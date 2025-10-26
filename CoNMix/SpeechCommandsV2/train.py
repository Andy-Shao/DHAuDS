import argparse
import numpy as np
import random
import os
import wandb
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip, Normalize

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.spdataset import SpeechCommandsV2
import CoNMix.lib.models as models
from CoNMix.lib.utils import time_shift, pad_trunc, Components, ExpandChannel, cal_norm
from AuT.lib.loss import CrossEntropyLabelSmooth

def lr_scheduler(optimizer: torch.optim.Optimizer, iter_num: int, max_iter: int, step:int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    wandb.log({'Train/learning_rate': param_group['lr']}, step=step)
    return optimizer

def build_optimizer(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in modelF.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * .1}]
    for k, v in modelB.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k, v in modelC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def load_models(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module]:
    modelF = models.ViT().to(device=args.device)
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).to(device=args.device)
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device=args.device)

    return modelF, modelB, modelC

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
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
    ap.add_argument('--lr_gamma', default=10, type=int)
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
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
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

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    max_ms=1000
    n_mels=81
    hop_length=200
    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=args.sample_rate),
        time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80),
        ExpandChannel(out_channel=3),
        Resize(size=(256, 256), antialias=False),
        RandomCrop(224),
        RandomHorizontalFlip()
    ]
    if args.normalized:
        print('calculate the train dataset mean and standard deviation')
        train_tf = Components(transforms=tf_array)
        train_dataset = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='training', download=True, 
            data_tf=train_tf
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
        train_mean, train_std = cal_norm(loader=train_loader)
        tf_array.append(Normalize(mean=train_mean, std=train_std))
        train_tf = Components(transforms=tf_array)
        train_dataset = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='training', download=True, 
            data_tf=train_tf
        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=args.sample_rate),
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80),
        ExpandChannel(out_channel=3),
        Resize((224, 224), antialias=False),
    ]
    if args.normalized:
        print('calculate the validation dataset mean and standard deviation')
        val_tf = Components(transforms=tf_array)
        val_dataset = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='validation', download=True,
            data_tf=val_tf
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
        val_mean, val_std = cal_norm(loader=val_loader)
        tf_array.append(Normalize(mean=val_mean, std=val_std))
    val_tf = Components(transforms=tf_array)
    val_dataset = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='validation', download=True,
        data_tf=val_tf
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(f'Total train dataset size: {len(train_dataset)}, batch size: {len(train_loader)}')
    print(f'Total validation dataset size: {len(val_dataset)}, batch size: {len(val_loader)}')

    modelF, modelB, modelC = load_models(args)
    store_model_structure_to_txt(model=modelF, output_path=os.path.join(args.output_path, 'modelF_structure.txt'))
    store_model_structure_to_txt(model=modelB, output_path=os.path.join(args.output_path, 'modelB_structure.txt'))
    store_model_structure_to_txt(model=modelC, output_path=os.path.join(args.output_path, 'modelC_structure.txt'))
    optimizer = build_optimizer(args, modelF=modelF, modelB=modelB, modelC=modelC)
    classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth, use_gpu=True).to(device=args.device)

    best_accuracy = 0.
    max_iter = args.max_epoch * len(train_loader)
    iter = 0
    interval = max_iter // args.interval
    for epoch in range(1, args.max_epoch+1):
        print(f'Epoch [{epoch}/{args.max_epoch}]')
        modelF.train(); modelB.train(); modelC.train()
        ttl_train_loss = 0.
        ttl_train_num = 0
        ttl_train_corr = 0
        print('Training....')
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = modelC(modelB(modelF(features)))
            loss = classifier_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_train_corr += (preds == labels).sum().cpu().item()
            ttl_train_loss += loss.cpu().item()
            ttl_train_num += labels.shape[0]
            features = None
            labels = None
            loss = None
            outputs = None

            learning_rate = optimizer.param_groups[0]['lr']
            if iter % interval == 0 or iter == max_iter-1:
                lr_scheduler(optimizer=optimizer, iter_num=iter, max_iter=max_iter, step=epoch)
            iter += 1

        modelF.eval()
        modelB.eval()
        modelC.eval()
        ttl_corr = 0
        ttl_size = 0
        print('Validating...')
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                outputs = modelC(modelB(modelF(features)))
                _, preds = torch.max(outputs, dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
        curr_accu = ttl_corr / ttl_size
        wandb_run.log(data={
            'Train/Loss': ttl_train_loss / ttl_train_num,
            'Train/Accu': ttl_train_corr/ttl_train_num,
            'Train/LR': learning_rate,
            'Val/Accu': curr_accu
        }, step=epoch, commit=True)
        if curr_accu > best_accuracy:
            best_accuracy = curr_accu
            best_modelF = modelF.state_dict()
            best_modelB = modelB.state_dict()
            best_modelC = modelC.state_dict()

            torch.save(best_modelF, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelF.pt'))
            torch.save(best_modelB, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelB.pt'))
            torch.save(best_modelC, os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelC.pt'))
        features = None
        labels = None
        outputs = None