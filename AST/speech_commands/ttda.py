import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms

from lib.utils import make_unless_exits, print_argparse, store_model_structure_to_txt
from lib import constants
from lib.acousticDataset import QUTNOISE, DEMAND
from lib.component import Components, Stereo2Mono, AudioPadding, ASTFeatureExt, DoNothing, time_shift
from lib.spdataset import SpeechCommandsV2
from lib.corruption import DynEN, WHN
from lib.dataset import mlt_store_to, mlt_load_from, MultiTFDataset, MergSet

def corrupt_data(args:argparse.Namespace) -> Dataset:
    if args.corruption_type == 'ENQ':
        if args.corruption_level == 'L1':
            snrs = [10, 1, 15]
        elif args.corruption_level == 'L2':
            snrs = [5, 1, 10]
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True,
            data_tf=Components(transforms=[
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                DynEN(noise_list=enq_noises(args), lsnr=snrs[0], rsnr=snrs[2], step=snrs[1])
            ])
        )
    elif args.corruption_type == 'END':
        if args.corruption_level == 'L1':
            snrs = [10, 1, 15]
        elif args.corruption_level == 'L2':
            snrs = [5, 1, 10]
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True,
            data_tf=Components(transforms=[
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                DynEN(noise_list=end_noises(args), lsnr=snrs[0], rsnr=snrs[2], step=snrs[1])
            ])
        )
    elif args.corruption_type == 'WHN':
        if args.corruption_level == 'L1':
            snrs = [10, 1, 15]
        elif args.corruption_level == 'L2':
            snrs = [5, 1, 10]
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True,
            data_tf=Components(transforms=[
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                WHN(lsnr=snrs[0], rsnr=snrs[2], step=snrs[1])
            ])
        )
    else:
        raise Exception('No support')
    return test_set

def inference(args:argparse.Namespace, ast:ASTForAudioClassification, data_loader:DataLoader) -> float:
    ast.eval()
    ttl_curr, ttl_size = 0., 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = ast(features).logits
            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_curr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
    return ttl_curr / ttl_size * 100.

def end_noises(args:argparse.Namespace, noise_modes:list[str] = ['DKITCHEN', 'NFIELD', 'OOFFICE', 'PRESTO', 'TCAR']) -> list[torch.Tensor]:
    noises = []
    print('Loading noise files...')
    demand_set = MergSet([DEMAND(root_path=args.noise_path, mode=md, include_rate=False) for md in noise_modes])
    for wavform in tqdm(demand_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def enq_noises(args:argparse.Namespace, noise_modes:list[str] = ['CAFE', 'HOME', 'STREET']) -> list[torch.Tensor]:
    background_path = args.noise_path
    noises = []
    print('Loading noise files...')
    qutnoise_set = MergSet([
        QUTNOISE(
            root_path=background_path, mode=md, include_rate=False,
            data_tf=Components(transforms=[
                transforms.Resample(orig_freq=48000, new_freq=args.sample_rate),
                Stereo2Mono()
            ])
        ) for md in noise_modes
    ])
    for wavform in tqdm(qutnoise_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def g_entropy(args:argparse.Namespace, outputs:torch.Tensor, q:float=.9) -> torch.Tensor:
    """
    " Generalized Entropy loss
    """
    if args.gent_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        gent_loss = (1 - torch.sum(torch.pow(softmax_outputs, exponent=q), dim=1)) / (q - 1)
        gent_loss = torch.mean(gent_loss)
        gent_loss = args.gent_rate * gent_loss
    else:
        gent_loss = torch.tensor(.0).to(args.device)
    return gent_loss

def entropy(args:argparse.Namespace, outputs:torch.Tensor, epsilon:float=1e-6) -> torch.Tensor:
    """
    " Entropy loss
    """
    if args.ent_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        ent_loss = - softmax_outputs * torch.log(softmax_outputs + epsilon)
        ent_loss = torch.mean(torch.sum(ent_loss, dim=1), dim=0)
        ent_loss = args.ent_rate * ent_loss
    else:
        ent_loss = torch.tensor(.0).to(args.device)
    return ent_loss

def nucnm(args:argparse.Namespace, outputs:torch.Tensor) -> torch.Tensor:
    """
    " Nuclear-norm Maximization loss with Frobenius Norm
    """
    if args.nucnm_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        nucnm_loss = - torch.mean(torch.sqrt(torch.sum(torch.pow(softmax_outputs,2),dim=0)))
        nucnm_loss = args.nucnm_rate * nucnm_loss
    else:
        nucnm_loss = torch.tensor(.0).to(args.device)
    return nucnm_loss

def build_model(args:argparse.Namespace) -> tuple[AutoFeatureExtractor, ASTForAudioClassification]:
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2").to(args.device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")

    return feature_extractor, ast

def lr_scheduler(optimizer: torch.optim.Optimizer, epoch:int, lr_cardinality:int, gamma=10, power=0.75, threshold=1) -> optim.Optimizer:
    if epoch >= lr_cardinality-threshold:
        return optimizer
    decay = (1 + gamma * epoch / lr_cardinality) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    return optimizer

def build_optimizer(args: argparse.Namespace, model:nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in model.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END', 'TST+PSH', 'DP+PSH'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AST'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'TTDA')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    fe, ast = build_model(args)
    store_model_structure_to_txt(
        model=ast, 
        output_path=os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.txt'))
    optimizer = build_optimizer(args=args, model=ast)
    test_set = corrupt_data(args)
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    mlt_store_to(
        dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name,
        data_tfs=[DoNothing()]
    )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')]
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, 
        pin_memory=True
    )
    adapt_set = mlt_load_from(root_path=dataset_root_path, index_file_name=index_file_name,)
    adapt_set = MultiTFDataset(
        dataset=adapt_set,
        tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
            ])
        ]
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )
    print('Pre-adaptation')
    accuracy = inference(args=args, ast=ast, data_loader=test_loader)
    print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(test_set)}')

    max_accu = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Adaptating...')
        ast.train()
        ttl_size = 0.
        ttl_loss = 0.
        ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.
        ttl_gent_loss = 0.
        for fs1, fs2, _ in tqdm(adapt_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1 = ast(fs1).logits
            os2 = ast(fs2).logits

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1) + entropy(args, os2)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os2, q=args.gent_q)

            loss = nucnm_loss + ent_loss + gent_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Inferencing...')
        accuracy = inference(args=args, ast=ast, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(adapt_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(
                ast.state_dict(), 
                os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt'))

        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/Entropy loss': ttl_ent_loss / ttl_size,
                'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
                'Adaptation/accuracy': accuracy,
                'Adaptation/LR': learning_rate,
                'Adaptation/max_accu': max_accu,
            }, step=epoch, commit=True
        )