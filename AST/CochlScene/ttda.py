import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from transformers import ASTForAudioClassification

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib import constants
from lib.component import Components, AudioPadding, DoNothing, ASTFeatureExt, time_shift
from lib.corruption import WHN, DynPSH
from lib.acousticDataset import CochlScene
from lib.dataset import GpuMultiTFDataset, mlt_load_from, mlt_store_to, MultiTFDataset
from AST.CochlScene.train import build_model, inference
from AST.speech_commands.ttda import nucnm, g_entropy, entropy, lr_scheduler, op_copy
from AST.lib.model import ASTClssifier

def build_optimizer(args: argparse.Namespace, model:nn.Module, classifier:nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in model.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * args.ast_lr_decay}]
    for k, v in classifier.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * args.clsf_lr_decay}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def load_weight(args:argparse.Namespace, model:ASTForAudioClassification, classifier:ASTClssifier, mode:str='origin') -> None:
    if mode == 'origin':
        ast_pth = args.ast_wght_pth
        clsf_pth = args.clsf_wght_pth
    elif mode == 'adaption':
        ast_pth = args.adpt_ast_wght_pth
        clsf_pth = args.adpt_clsf_wght_pth
    else: 
        raise Exception('No Support')
    model.load_state_dict(state_dict=torch.load(f=ast_pth, weights_only=True))
    classifier.load_state_dict(state_dict=torch.load(f=clsf_pth, weights_only=True))

def corrupt_data(args:argparse.Namespace) -> Dataset:
    if args.corruption_level == 'L1':
        snrs = [10, 1, 15]
        steps = [0, 3]
        speeds = [.05, .05, .1]
    elif args.corruption_level == 'L2':
        snrs = [5, 1, 10]
        steps = [2, 5]
        speeds = [.1, .01, .2]
    if args.corruption_type == 'WHNP':
        test_set = CochlScene(
            root_path=args.dataset_root_path, mode='test', include_rate=False,
            data_tf=Components(transforms=[
                transforms.Resample(orig_freq=args.org_sample_rate, new_freq=args.sample_rate),
                AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                WHN(lsnr=snrs[0], rsnr=snrs[2], step=snrs[1]),
            ])
        )
        # test_set = GpuMultiTFDataset(
        #     dataset=test_set, device=args.device, maintain_cpu=True,
        #     tfs=[
        #         DynPSH(sample_rate=args.sample_rate, min_steps=steps[0], max_steps=steps[1], is_bidirection=True)
        #     ]
        # )
    return test_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='CochlScene', choices=['CochlScene'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHNP', 'WHNT', 'HVP', 'HVT'])
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
    ap.add_argument('--ast_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--ast_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'CochlScene':
        args.class_num = 13
        args.sample_rate = 16000
        args.org_sample_rate = 44100
        if not os.path.exists(args.dataset_root_path):
            os.makedirs(args.dataset_root_path)
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
    
    fe, ast, clsf = build_model(args)
    load_weight(args=args, model=ast, classifier=clsf)
    optimizer = build_optimizer(args=args, model=ast, classifier=clsf)
    param_no = count_ttl_params(ast) + count_ttl_params(clsf)
    print(f'Param No. is: {param_no}')
    test_set = corrupt_data(args)
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    # mlt_store_to(
    #     dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name,
    #     data_tfs=[DoNothing()]
    # )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')]
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=30, shuffle=False, drop_last=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    adapt_set = mlt_load_from(root_path=dataset_root_path, index_file_name=index_file_name,)
    adapt_set = MultiTFDataset(
        dataset=adapt_set,
        tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
            ]),
            # Components(transforms=[
            #     time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
            #     ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')
            # ])
        ]
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )
    print('Pre-adaptation')
    accuracy = inference(args=args, model=ast, classifier=clsf, data_loader=test_loader)
    print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(test_set)}')

    max_accu = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Adaptating...')
        ast.train(); clsf.train()
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.; ttl_gent_loss = 0.
        for fs1, _ in tqdm(adapt_loader):
            fs1 = fs1.to(args.device)

            optimizer.zero_grad()
            os1 = clsf(ast(fs1).logits)
            # os2 = clsf(ast(fs2).logits)

            # nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            # ent_loss = entropy(args, os1) + entropy(args, os2)
            # gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os2, q=args.gent_q)
            nucnm_loss = nucnm(args, os1)
            ent_loss = entropy(args, os1)
            gent_loss = g_entropy(args, os1, q=args.gent_q)

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
        accuracy = inference(args=args, model=ast, classifier=clsf, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(adapt_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(
                ast.state_dict(), 
                os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-ast-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt'))
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-clsf-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt'))
                
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