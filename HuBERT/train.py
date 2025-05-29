import argparse
from tqdm import tqdm
import os
import wandb
import numpy as np
import random

import torchaudio
from torchvision.transforms import Compose
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from HuBERT.lib.model import Classifier
from lib.utils import print_argparse, store_model_structure_to_txt, make_unless_exits
from lib.component import AudioPadding, ReduceChannel, time_shift
from lib.dataset import dataset_tag
from lib.spdataset import SpeechCommandsV2

def build_model(args:argparse.Namespace) -> tuple[torchaudio.models.Wav2Vec2Model, Classifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model().to(device=args.device)
    classifier = Classifier(class_num=args.class_num, embed_size=768).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default='2025')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.class_num = 35
    args.sample_rate = 16000
    args.output_path = os.path.join(args.output_path, 'HuBERT', args.dataset)

    torch.backends.cudnn.benchmark == True
    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    print_argparse(args=args)
    ############################################################

    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)

    wandb_run = wandb.init(
        project='NTTA-Train', name=f'HuB-{dataset_tag(dataset=args.dataset)}', mode='online' if args.wandb else 'disabled', config=args, 
        tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    max_length = args.sample_rate
    train_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, folder_in_archive='speech_commands_v0.02', mode='training', 
        data_tf=Compose(transforms=[
            time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
            AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=True),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    val_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, folder_in_archive='speech_commands_v0.02', mode='validation', 
        data_tf=Compose(transforms=[
            AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=False),
            ReduceChannel()
        ])
    )
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    hubert, clsModel = build_model(args=args)
    store_model_structure_to_txt(model=hubert, output_path=os.path.join(args.output_path, 'hubert.txt'))
    store_model_structure_to_txt(model=clsModel, output_path=os.path.join(args.output_path, 'clsModel.txt'))
    optimizer = optim.SGD(lr=args.lr, params=clsModel.parameters())
    loss_fn = nn.CrossEntropyLoss().to(device=args.device)

    hubert.eval()
    clsModel.train()
    ttl_corr = 0.
    ttl_size = 0.
    train_loss = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            hiddent_fs = hubert.extract_features(features)[0]
            hiddent_fs = hiddent_fs[-1]
            outputs = clsModel(hiddent_fs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            train_loss += loss.cpu().item()
        train_accu = ttl_corr/ttl_size * 100.
        print(f'Training accuracy is: {train_accu:.4f}%, sample size is: {len(train_set)}')

        print('Validating...')
        hubert.eval()
        clsModel.eval()
        max_accu = 0.
        ttl_corr = 0.
        ttl_size = 0.
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            with torch.no_grad():
                hiddent_fs = hubert.extract_features(features)[0]
                hiddent_fs = hiddent_fs[-1]
                outputs = clsModel(hiddent_fs)
            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
        val_accu = ttl_corr / ttl_size * 100.
        print(f'Validation accuracy is: {val_accu:.4f}%, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/loss': train_loss / len(train_loader),
            'Train/accuracy': train_accu,
            'Val/accuracy': val_accu
        }, step=epoch, commit=True, sync=True)

        if max_accu <= val_accu:
            max_accu == val_accu
            torch.save(obj=clsModel.state_dict(), f=os.path.join(args.output_path, 'clsModel.pt'))
            torch.save(obj=hubert.state_dict(), f=os.path.join(args.output_path, 'hubert.pt'))