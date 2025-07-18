import argparse
import os
import random
import wandb
import numpy as np
from tqdm import tqdm

import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchaudio import transforms
from transformers import AutoFeatureExtractor, ASTForAudioClassification

from lib.utils import make_unless_exits, print_argparse, ConfigDict, count_ttl_params, store_model_structure_to_txt
from lib import constants
from lib.spdataset import AudioMNIST
from lib.component import Components, AudioPadding, ASTFeatureExt, time_shift
from AST.lib.model import ASTClssifier
from AST.speech_commands.ttda import op_copy, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth

def inference(args:argparse.Namespace, model:nn.Module, classifier:nn.Module, data_loader:DataLoader) -> float:
    model.eval(); classifier.eval()
    ttl_curr, ttl_size = 0., 0
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = classifier(model(features).logits)
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_curr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_curr / ttl_size * 100.

def build_optimizer(args: argparse.Namespace, model:nn.Module, classifier:nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in model.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k, v in classifier.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def build_model(args:argparse.Namespace) -> tuple[AutoFeatureExtractor, ASTForAudioClassification, ASTClssifier]:
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2").to(args.device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")

    cfg = ConfigDict()
    cfg.embedding = ConfigDict()
    cfg.embedding['embed_size'] = ast.config.num_labels
    cfg.classifier = ConfigDict()
    cfg.classifier['class_num'] = args.class_num
    classifier = ASTClssifier(config=cfg).to(device=args.device)

    return feature_extractor, ast, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='AudioMNIST', choices=['AudioMNIST'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'AudioMNIST':
        args.class_num = 10
        args.sample_rate = 16000
        args.org_sample_rate = 48000
        if not os.path.exists(args.dataset_root_path):
            os.makedirs(args.dataset_root_path)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AST'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Train')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    fe, ast, clsf = build_model(args)
    param_no = count_ttl_params(ast) + count_ttl_params(clsf)
    print(f'Param No. is: {param_no}')
    store_model_structure_to_txt(model=ast, output_path=os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-ast{args.file_suffix}.txt'))
    store_model_structure_to_txt(model=clsf, output_path=os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-clsf{args.file_suffix}.txt'))

    train_set = AudioMNIST(
        data_paths=AudioMNIST.default_splits(root_path=args.dataset_root_path, mode='train'), include_rate=False, 
        data_trainsforms=Components(transforms=[
            transforms.Resample(orig_freq=args.org_sample_rate, new_freq=args.sample_rate),
            AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=True),
            time_shift(shift_limit=.17, is_random=True, is_bidirection=True), 
            ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, 
        num_workers=args.num_workers
    )

    val_set = AudioMNIST(
        data_paths=AudioMNIST.default_splits(root_path=args.dataset_root_path, mode='validate'), include_rate=False, 
        data_trainsforms=Components(transforms=[
            transforms.Resample(orig_freq=args.org_sample_rate, new_freq=args.sample_rate),
            AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
            ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
        ])
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=4
    )

    optimizer = build_optimizer(args=args, model=ast, classifier=clsf)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Training...')
        ast.train(); clsf.train()
        ttl_curr, ttl_size, ttl_loss = 0., 0., 0.
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsf(ast(features).logits)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_curr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
            ttl_loss += loss.cpu().item()
        train_accu = ttl_curr / ttl_size * 100.
        print(f'Training accuracy is: {train_accu:.4f}%, sample size is: {len(train_set)}')

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Validating...')
        val_accu = inference(args=args, model=ast, classifier=clsf, data_loader=val_loader)
        if max_accu <= val_accu:
            max_accu = val_accu
            torch.save(obj=ast.state_dict(), f=os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-ast{args.file_suffix}.pt'))
            torch.save(obj=clsf.state_dict(), f=os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-clsf{args.file_suffix}.pt'))
        print(f'Validation accuracy is: {val_accu:.4f}%, sample size is: {len(val_set)}')

        wandb_run.log({
            'Loss/train_loss': ttl_loss / len(train_loader),
            'Loss/learning_rate': learning_rate,
            'Accu/train_accu': train_accu,
            'Accu/val_accu': val_accu,
        }, step=epoch, commit=True)