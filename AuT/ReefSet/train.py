import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt, indexes2oneHot
from lib.acousticDataset import ReefSet
from lib.component import OneHot2Index, Components, AudioPadding, AudioClip, time_shift, AmplitudeToDB
from lib.component import FrequenceTokenTransformer
from lib.lr_utils import build_optimizer, lr_scheduler
from lib.dataset import MultiTFDataset
from lib.corruption import end_noises, DynEN
from AuT.lib.model import FCETransform, AudioClassifier
from AuT.lib.config import AuT_base
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(args:argparse.Namespace, aut:FCETransform, clsf:AudioClassifier, data_loader:DataLoader) -> float:
    aut.eval(); clsf.eval()
    for idx, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs, _ = clsf(aut(features)[0])

        if idx == 0:
            y_true = indexes2oneHot(labels=labels, class_num=args.class_num)
            y_score = outputs.detach().cpu()
        else:
            y_true = torch.cat([y_true, indexes2oneHot(labels=labels, class_num=args.class_num)], dim=0)
            y_score = torch.cat([y_score, outputs.detach().cpu()], dim=0)
    val_roc_auc = roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro')
    return val_roc_auc

def build_model(args:argparse.Namespace) -> tuple[FCETransform, AudioClassifier]:
    cfg = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    cfg.embedding.in_shape = [args.n_mels, args.target_length]
    cfg.embedding.width = 128
    cfg.embedding.num_layers = [6, 8]
    cfg.embedding.embed_num = 24
    cfg.classifier.in_embed_num = 26
    aut = FCETransform(config=cfg).to(device=args.device)
    clsf = AudioClassifier(config=cfg).to(device=args.device)
    return aut, clsf

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
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
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
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

    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195
    train_set = ReefSet(
        root_path=args.dataset_root_path, mode='train', include_rate=False, label_mode='single', 
        label_tf=OneHot2Index(),
    )
    train_set = MultiTFDataset(
        dataset=train_set,
        tfs=[
            Components(transforms=[
                AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True),
                AudioClip(max_length=args.audio_length, is_random=True),
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    mel_scale=mel_scale, n_mels=args.n_mels
                ), # 80 x 195
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                DynEN(
                    noise_list=end_noises(
                        end_path=args.background_path, sample_rate=args.sample_rate, 
                        noise_modes=['DWASHING']
                    ), lsnr=55, rsnr=55, step=0
                ),
                AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True),
                AudioClip(max_length=args.audio_length, is_random=True),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    mel_scale=mel_scale, n_mels=args.n_mels
                ), # 80 x 195
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        ]
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )

    val_set = ReefSet(
        root_path=args.dataset_root_path, mode='test', include_rate=False, label_mode='single',
        label_tf=OneHot2Index(),
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, is_random=False, mode='head'),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                mel_scale=mel_scale, n_mels=args.n_mels
            ), # 80 x 195
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )

    aut, clsf = build_model(args=args)
    store_model_structure_to_txt(model=aut, output_path=os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}.txt'))
    store_model_structure_to_txt(model=clsf, output_path=os.path.join(args.output_path, f'clsf-{constants.dataset_dic[args.dataset]}.txt'))
    optimizer = build_optimizer(lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_roc_auc = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        aut.train(); clsf.train()
        train_loss = 0.
        for idx, fs in tqdm(enumerate(train_loader), total=len(train_loader)):
            labels = fs[-1].to(args.device)

            optimizer.zero_grad()
            for i in range(len(fs) - 1):
                features = fs[i].to(args.device)
                outputs, _ = clsf(aut(features)[0])
                if i == 0:
                    loss = loss_fn(outputs, labels)
                else:
                    loss += loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if idx == 0:
                y_true = indexes2oneHot(labels=labels, class_num=args.class_num)
                y_score = outputs.detach().cpu()
            else:
                y_true = torch.cat([y_true, indexes2oneHot(labels=labels, class_num=args.class_num)], dim=0)
                y_score = torch.cat([y_score, outputs.detach().cpu()], dim=0)
            train_loss += loss.cpu().item()
        train_roc_auc = roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro')
        print(f'Training Mean ROC-AUC is: {train_roc_auc:.4f}, sample size is: {len(train_set)}')
        y_true = None; y_score = None

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        print('Validating...')
        val_roc_auc = inference(args, aut=aut, clsf=clsf, data_loader=val_loader)
        print(f'Validation Mean ROC-AUC is: {val_roc_auc:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/ROC-AUC': train_roc_auc,
            'Train/LR': learning_rate,
            'Val/ROC-AUC': val_roc_auc
        }, step=epoch, commit=True)

        if max_roc_auc <= val_roc_auc:
            max_roc_auc == val_roc_auc
            torch.save(obj=clsf.state_dict(), f=os.path.join(args.output_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt'))
            torch.save(obj=aut.state_dict(), f=os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}.pt'))