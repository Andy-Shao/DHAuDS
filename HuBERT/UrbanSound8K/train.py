import argparse
import numpy as np
import random 
import os
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.models import Wav2Vec2Model

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt, ConfigDict
from lib.enDataset import UrbanSound8K
from lib.component import Components, AudioPadding, Stereo2Mono, ReduceChannel, time_shift, AudioClip
from lib.lr_utils import build_optimizer, lr_scheduler
from AuT.lib.model import AudioClassifier
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(args:argparse.Namespace, hubert:nn.Module, clsModel:nn.Module, data_loader:DataLoader):
    hubert.eval(); clsModel.eval()
    for idx, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs, _ = clsModel(hubert(features)[0])
        _, preds = torch.max(outputs.detach().cpu(), dim=1)
        
        if idx == 0:
            y_true = labels.detach().cpu()
            y_pred = preds
        else:
            y_true = torch.cat([y_true, labels.detach().cpu()], dim=0)
            y_pred = torch.cat([y_pred, preds], dim=0)
    val_f1 = f1_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average='macro')
    return val_f1

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[Wav2Vec2Model, AudioClassifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    cfg = ConfigDict()
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = args.class_num
    cfg.classifier['extend_size'] = 2048
    cfg.classifier['convergent_size'] = 256
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = bundle._params['encoder_embed_dim']
    classifier = AudioClassifier(config=cfg).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
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
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 16000
        args.audio_length = int(4 * 16000)
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

    train_set = UrbanSound8K(
        root_path=args.dataset_root_path, include_rate=False, folds=[1, 2, 3, 4, 5, 6, 7], sample_rate=args.sample_rate,
        data_tf=Components(transforms=[
            Stereo2Mono(),
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True),
            AudioClip(max_length=args.audio_length, is_random=True),
            time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
            ReduceChannel()
        ])
    )
    val_set = UrbanSound8K(
        root_path=args.dataset_root_path, include_rate=False, folds=[8, 9, 10], sample_rate=args.sample_rate,
        data_tf=Components(transforms=[
            Stereo2Mono(),
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=True), 
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
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

    max_f1 = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        hubert.train(); clsModel.train()
        train_loss = 0.
        for idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs, _ = clsModel(hubert(features)[0])
            loss = loss_fn(outputs, labels)
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
        y_true = None; y_score = None

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Validating...')
        val_f1 = inference(args=args, hubert=hubert, clsModel=clsModel, data_loader=val_loader)
        print(f'Validation F1 score is: {val_f1:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/F1_score': train_f1,
            'Train/LR': learning_rate,
            'Val/F1_score': val_f1
        }, step=epoch, commit=True)

        if max_f1 <= val_f1:
            max_f1 == val_f1
            torch.save(obj=clsModel.state_dict(), f=os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt'))
            torch.save(obj=hubert.state_dict(), f=os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt'))