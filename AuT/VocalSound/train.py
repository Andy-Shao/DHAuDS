import argparse
import numpy as np
import random
import wandb
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.spdataset import VocalSound
from lib.component import Components, AudioPadding, AudioClip, time_shift, AmplitudeToDB
from lib.component import FrequenceTokenTransformer, MelSpectrogramPadding
from lib.lr_utils import build_optimizer, lr_scheduler
from AuT.lib.model import FCETransform, FCEClassifier
from AuT.lib.config import AuT_base
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(args:argparse.Namespace, aut:FCETransform, clsf:FCEClassifier, data_loader:DataLoader) -> float:
    aut.eval(); clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs = clsf(aut(features)[1])
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size

def build_model(args:argparse.Namespace) -> tuple[FCETransform, FCEClassifier]:
    config = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    config.embedding.in_shape = [args.n_mels, args.target_length]
    config.embedding.num_layers = [6, 4, 8]
    config.embedding.width = 64
    config.embedding.embed_num = 65
    config.classifier.in_embed_num = 2
    clsmodel = FCEClassifier(config=config).to(device=args.device)
    auTmodel = FCETransform(config=config).to(device=args.device)

    return auTmodel, clsmodel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--aut_lr_decay', type=float, default=1.)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.)
    ap.add_argument('--smooth', type=float, default=.1)

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    args.arch = 'AMAuT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'train')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    make_unless_exits(args.output_path)

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    train_set = VocalSound(
        root_path=args.dataset_root_path, mode='train', include_rate=False, version='16k',
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True),
            AudioClip(max_length=args.audio_length, is_random=True),
            time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                mel_scale=mel_scale, n_mels=args.n_mels
            ), # 80 x 1039
            AmplitudeToDB(top_db=80., max_out=2.),
            MelSpectrogramPadding(target_length=args.audio_length), 
            FrequenceTokenTransformer()
        ])
    )
    val_set = VocalSound(
        root_path=args.dataset_root_path, mode='validation', include_rate=False, version='16k',
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, 
                mel_scale=mel_scale, n_mels=args.n_mels
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            MelSpectrogramPadding(target_length=args.audio_length),
            FrequenceTokenTransformer()
        ])
    )
    
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    aut, clsf = build_model(args)
    store_model_structure_to_txt(model=aut, output_path=os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}.txt'))
    store_model_structure_to_txt(model=clsf, output_path=os.path.join(args.output_path, f'clsf-{constants.dataset_dic[args.dataset]}.txt'))
    optimizer = build_optimizer(lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        aut.train(); clsf.train()
        ttl_corr = 0.; ttl_size = 0.; train_loss = 0.
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsf(aut(features)[1])
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            train_loss += loss.cpu().item()
        train_accu = ttl_corr/ttl_size
        print(f'Training accuracy is: {train_accu:.4f}, sample size is: {len(train_set)}')

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma)

        print('Validating...')
        val_accu = inference(args=args, aut=aut, clsf=clsf, data_loader=val_loader)
        print(f'Validation accuracy is: {val_accu:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/Accu': train_accu,
            'Train/LR': learning_rate,
            'Val/Accu': val_accu
        }, step=epoch, commit=True)

        if max_accu <= val_accu:
            max_accu == val_accu
            torch.save(obj=clsf.state_dict(), f=os.path.join(args.output_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt'))
            torch.save(obj=aut.state_dict(), f=os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}.pt'))