import argparse
import os
import wandb
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip, Normalize

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.utils import indexes2oneHot
from lib.component import Components, AudioPadding, OneHot2Index, AudioClip
from lib.acousticDataset import ReefSet
from CoNMix.lib.utils import time_shift, ExpandChannel, cal_norm
from CoNMix.SpeechCommandsV2.train import load_models, build_optimizer, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, data_loader: DataLoader, device='cpu') -> float:
    modelF.eval(); modelB.eval(); modelC.eval()
    for idx, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            outputs = modelC(modelB(modelF(features)))

        if idx == 0:
            y_true = indexes2oneHot(labels=labels, class_num=args.class_num)
            y_score = outputs.detach().cpu()
        else:
            y_true = torch.cat([y_true, indexes2oneHot(labels=labels, class_num=args.class_num)], dim=0)
            y_score = torch.cat([y_score, outputs.detach().cpu()], dim=0)
    return roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro')
        

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--normalized', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--aut_lr_decay', type=float, default=1.)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.)
    ap.add_argument('--smooth', type=float, default=.1)

    # CoNMix hyperparameters
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    args.arch = 'CoNMix'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'train')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    make_unless_exits(args.output_path)

    wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    n_mels=123
    hop_length=250
    tf_array = [
        AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
        AudioClip(max_length=args.audio_length, is_random=True),
        time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize(size=(256, 256), antialias=False),
        RandomCrop(224),
        RandomHorizontalFlip()
    ]
    if args.normalized:
        print('calculate the train dataset mean and standard deviation')
        train_dataset = ReefSet(
            root_path=args.dataset_root_path, mode='train', include_rate=False, 
            data_tf=Components(transforms=tf_array), label_tf=OneHot2Index()
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
        train_mean, train_std = cal_norm(loader=train_loader)
        tf_array.append(Normalize(mean=train_mean, std=train_std))
    train_dataset = ReefSet(
        root_path=args.dataset_root_path, mode='train', include_rate=False, 
        data_tf=Components(transforms=tf_array), label_tf=OneHot2Index()
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = [
        AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
        AudioClip(max_length=args.audio_length, is_random=False, mode='head'),
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize((224, 224), antialias=False),
    ]
    if args.normalized:
        print('calculate the validation dataset mean and standard deviation')
        val_dataset = ReefSet(
            root_path=args.dataset_root_path, mode='test', include_rate=False,
            data_tf=Components(transforms=tf_array), label_tf=OneHot2Index()
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
        val_mean, val_std = cal_norm(loader=val_loader)
        tf_array.append(Normalize(mean=val_mean, std=val_std))
    val_dataset = ReefSet(
        root_path=args.dataset_root_path, mode='test', include_rate=False,
        data_tf=Components(transforms=tf_array), label_tf=OneHot2Index()
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(f'Total train dataset size: {len(train_dataset)}, batch number: {len(train_loader)}')
    print(f'Total validation dataset size: {len(val_dataset)}, batch number: {len(val_loader)}')

    modelF, modelB, modelC = load_models(args)
    store_model_structure_to_txt(model=modelF, output_path=os.path.join(args.output_path, 'modelF_structure.txt'))
    store_model_structure_to_txt(model=modelB, output_path=os.path.join(args.output_path, 'modelB_structure.txt'))
    store_model_structure_to_txt(model=modelC, output_path=os.path.join(args.output_path, 'modelC_structure.txt'))
    optimizer = build_optimizer(args, modelF=modelF, modelB=modelB, modelC=modelC)
    classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth, use_gpu=True).to(device=args.device)

    max_roc_auc = 0.
    max_iter = args.max_epoch * len(train_loader)
    iter = 0
    interval = max_iter // args.interval
    for epoch in range(1, args.max_epoch+1):
        print(f'Epoch [{epoch}/{args.max_epoch}]')
        modelF.train(); modelB.train(); modelC.train()
        print('Training....')
        train_loss = 0.
        for idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            features, labels = features.to(args.device), labels.to(args.device)

            outputs = modelC(modelB(modelF(features)))
            loss = classifier_loss(outputs, labels)
            optimizer.zero_grad()
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
        print(f'Training Mean ROC-AUC is: {train_roc_auc:.4f}, sample size is: {len(train_dataset)}')
        y_true = None; y_score = None

        learning_rate = optimizer.param_groups[0]['lr']
        if iter % interval == 0 or iter == max_iter-1:
            lr_scheduler(optimizer=optimizer, iter_num=iter, max_iter=max_iter, step=epoch)
        iter += 1

        print('Validating...')
        val_roc_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=val_loader, device=args.device)
        print(f'Validation Mean ROC-AUC is: {val_roc_auc:.4f}, sample size is: {len(val_dataset)}')

        wandb.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/ROC-AUC': train_roc_auc,
            'Val/ROC-AUC': val_roc_auc
        }, step=epoch, commit=True)
        if max_roc_auc <= val_roc_auc:
            max_roc_auc == val_roc_auc
            best_modelF = modelF.state_dict()
            best_modelB = modelB.state_dict()
            best_modelC = modelC.state_dict()

            torch.save(best_modelF, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelF.pt'))
            torch.save(best_modelB, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelB.pt'))
            torch.save(best_modelC, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelC.pt'))
        features = None
        labels = None
        outputs = None