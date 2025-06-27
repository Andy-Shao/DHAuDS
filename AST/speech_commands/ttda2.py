import argparse
import os
import numpy as np
import random
import wandb

from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
from torch import optim
from torch import nn

from lib.utils import make_unless_exits, print_argparse
from lib import constants

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
        name=f'{args.arch}-{constants.dataset_dic[args.dataset]}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    fe, ast = build_model(args)