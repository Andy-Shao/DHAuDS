import argparse
from tqdm import tqdm
import os
import wandb
import numpy as np
import random

import torchaudio
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader

from AuT.lib.model import AudioClassifier
from AuT.speech_commands.train import build_optimizer, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth
from lib.utils import print_argparse, store_model_structure_to_txt, make_unless_exits, ConfigDict
from lib.component import AudioPadding, ReduceChannel, time_shift
from lib.spdataset import SpeechCommandsV2, SpeechCommandsV1
from lib import constants

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[torchaudio.models.Wav2Vec2Model, AudioClassifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    cfg = ConfigDict()
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = args.class_num
    cfg.classifier.extend_size = 2048
    cfg.classifier.convergent_size = 256
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = bundle._params['encoder_embed_dim']
    classifier = AudioClassifier(config=cfg).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default='2025')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    elif args.dataset == 'SpeechCommandsV1':
        args.class_num = 30
    else:
        raise Exception('No support!')
    args.sample_rate = 16000
    arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'train')

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
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'HuB-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    max_length = args.sample_rate
    if args.dataset == 'SpeechCommandsV2':
        train_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='training', 
            data_tf=Compose(transforms=[
                AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=True),
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                ReduceChannel()
            ])
        )
    elif args.dataset == 'SpeechCommandsV1':
        train_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='train', include_rate=False,
            data_tfs=Compose(transforms=[
                AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=True),
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                ReduceChannel()
            ])
        )

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    if args.dataset == 'SpeechCommandsV2':
        val_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='validation', 
            data_tf=Compose(transforms=[
                AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=False),
                ReduceChannel()
            ])
        )
    elif args.dataset == 'SpeechCommandsV1':
        val_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='validation', include_rate=False,
            data_tfs=Compose(transforms=[
                AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=False),
                ReduceChannel()
            ])
        )
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    hubert, clsModel = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    store_model_structure_to_txt(model=hubert, output_path=os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}.txt'))
    store_model_structure_to_txt(model=clsModel, output_path=os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}.txt'))
    optimizer = build_optimizer(args=args, auT=hubert, auC=clsModel)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    hubert.train()
    clsModel.train()
    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        ttl_corr = 0.
        ttl_size = 0.
        train_loss = 0.
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs, _ = clsModel(hubert(features)[0])
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            train_loss += loss.cpu().item()
        train_accu = ttl_corr/ttl_size * 100.
        print(f'Training accuracy is: {train_accu:.4f}%, sample size is: {len(train_set)}')

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma)

        print('Validating...')
        hubert.eval()
        clsModel.eval()
        ttl_corr = 0.
        ttl_size = 0.
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            with torch.no_grad():
                outputs, _ = clsModel(hubert(features)[0])
            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
        val_accu = ttl_corr / ttl_size * 100.
        print(f'Validation accuracy is: {val_accu:.4f}%, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/Accu': train_accu,
            'Train/LR': learning_rate,
            'Val/Accu': val_accu
        }, step=epoch, commit=True)

        if max_accu <= val_accu:
            max_accu == val_accu
            torch.save(obj=clsModel.state_dict(), f=os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt'))
            torch.save(obj=hubert.state_dict(), f=os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt'))