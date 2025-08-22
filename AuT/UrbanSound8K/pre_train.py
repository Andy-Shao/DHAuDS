import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.acousticDataset import CochlScene
from lib.component import Components, AudioPadding, AudioClip, AmplitudeToDB, FrequenceTokenTransformer
from lib.component import time_shift
from lib.dataset import MultiTFDataset
from lib.lr_utils import build_optimizer, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth
from AuT.UrbanSound8K.train import build_model, inference

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='CochlScene', choices=['CochlScene'])
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
    if args.dataset == 'CochlScene':
        args.class_num = 13
        args.sample_rate = 44100
        args.audio_length = int(4 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    args.arch = 'AMAuT'
    target_dataset = 'UrbanSound8K'
    args.output_path = os.path.join(args.output_path, target_dataset, args.arch, 'PreTrain')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    make_unless_exits(args.output_path)

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'Pre-{constants.architecture_dic[args.arch]}-{constants.dataset_dic[target_dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    args.n_mels=80
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=589
    train_set = MultiTFDataset(
        dataset=CochlScene(root_path=args.dataset_root_path, mode='train', include_rate=False),
        tfs=[
            Components(transforms=[
                AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True),
                AudioClip(max_length=args.audio_length, is_random=True),
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    mel_scale=mel_scale, n_mels=args.n_mels
                ), # 80 x 589
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        ]
    )
    val_set = CochlScene(
        root_path=args.dataset_root_path, mode='validation', include_rate=False,
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                mel_scale=mel_scale, n_mels=args.n_mels
            ), # 80 x 589
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )

    aut, clsf = build_model(args)
    store_model_structure_to_txt(model=aut, output_path=os.path.join(args.output_path, f'pre-aut-{constants.dataset_dic[args.dataset]}.txt'))
    store_model_structure_to_txt(model=clsf, output_path=os.path.join(args.output_path, f'pre-clsf-{constants.dataset_dic[args.dataset]}.txt'))
    optimizer = build_optimizer(lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_f1 = 0.
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

            _, preds = torch.max(outputs.detach().cpu(), dim=1)
            
            if idx == 0:
                y_true = labels.detach().cpu()
                y_pred = preds
            else:
                y_true = torch.cat([y_true, labels.detach().cpu()], dim=0)
                y_pred = torch.cat([y_pred, preds], dim=0)
            train_loss += loss.cpu().item()
        train_f1 = f1_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average='macro')
        print(f'Training F1 score is: {train_f1:.4f}, sample size is: {len(train_set)}')
        y_true = None; y_pred = None

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        print('Validating...')
        val_f1 = inference(args, aut=aut, clsf=clsf, data_loader=val_loader)
        print(f'Validation F1 score is: {val_f1:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/F1_score': train_f1,
            'Train/LR': learning_rate,
            'Val/F1_score': val_f1
        }, step=epoch, commit=True)

        if max_f1 <= val_f1:
            max_f1 == val_f1
            torch.save(obj=clsf.state_dict(), f=os.path.join(args.output_path, f'pre-clsf-{constants.dataset_dic[args.dataset]}.pt'))
            torch.save(obj=aut.state_dict(), f=os.path.join(args.output_path, f'pre-aut-{constants.dataset_dic[args.dataset]}.pt'))