import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from lib.utils import make_unless_exits, print_argparse, ConfigDict, store_model_structure_to_txt
from lib import constants
from lib.spdataset import VocalSound
from lib.component import Components, AudioPadding, time_shift, ReduceChannel, AudioClip
from lib.lr_utils import build_optimizer, lr_scheduler
from AuT.lib.model import FCEClassifier
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(args:argparse.Namespace, hubert:nn.Module, clsModel:nn.Module, data_loader:DataLoader):
    hubert.eval(); clsModel.eval()
    ttl_corr = 0.; ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = clsModel(hubert(features)[0][:, :2, :])
        ttl_size += labels.shape[0]
        _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
    return ttl_corr / ttl_size * 100.

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[torchaudio.models.Wav2Vec2Model, FCEClassifier]:
    if args.model_level == 'base':
        bundle = torchaudio.pipelines.HUBERT_BASE
    elif args.model_level == 'large':
        bundle = torchaudio.pipelines.HUBERT_LARGE
    elif args.model_level == 'x-large':
        bundle = torchaudio.pipelines.HUBERT_XLARGE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    cfg = ConfigDict()
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = args.class_num
    cfg.classifier['in_embed_num'] = 2
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = bundle._params['encoder_embed_dim']
    classifier = FCEClassifier(config=cfg).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
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
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'train')

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
        name=f'{constants.architecture_dic[args.arch]}-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    train_set = VocalSound(
        root_path=args.dataset_root_path, mode='train', include_rate=False, version='16k', 
        data_tf=Components(transforms=[
            AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=True),
            AudioClip(max_length=10*args.sample_rate, is_random=True),
            time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    val_set = VocalSound(
        root_path=args.dataset_root_path, mode='validation', include_rate=False, version='16k', 
        data_tf=Components(transforms=[
            AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=10*args.sample_rate, is_random=False, mode='head'),
            ReduceChannel()
        ])
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    hubert, clsModel = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    store_model_structure_to_txt(model=hubert, output_path=os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}.txt'))
    store_model_structure_to_txt(model=clsModel, output_path=os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}.txt'))
    optimizer = build_optimizer(lr=args.lr, auT=hubert, auC=clsModel, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        hubert.train(); clsModel.train()
        ttl_corr = 0.; ttl_size = 0.; train_loss = 0.
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsModel(hubert(features)[0][:, :2, :])
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
        val_accu = inference(args=args, hubert=hubert, clsModel=clsModel, data_loader=val_loader)
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