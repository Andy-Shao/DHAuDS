import argparse
import numpy as np
import random
import wandb
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.dataset import MultiTFDataset, mlt_load_from, mlt_store_to, batch_store_to
from lib.spdataset import SpeechCommandsV2
from lib.component import Components, AudioPadding, ReduceChannel, time_shift, DoNothing
from lib.lr_utils import build_optimizer, lr_scheduler
from lib.loss import nucnm, g_entropy, entropy
from lib.corruption import DynTST
from HuBERT.VocalSound.ttda import load_weigth
from HuBERT.SpeechCommandsV2.train import build_model, inference

def sc_corruption_set(args:argparse.Namespace) -> tuple[Dataset, Dataset]:
    from HuBERT.VocalSound.ttda import corrupt_data as corrupt_data_tmp
    if args.corruption_type == 'TST':
        if args.corruption_level == 'L1':
            rates = [.06, .01, .1]
        elif args.corruption_level == 'L2':
            rates = [.08, .01, .12]
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True,
            data_tf=Components(transforms=[
                DynTST(min_rate=rates[0], step=rates[1], max_rate=rates[2], is_bidirection=False),
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
            ])
        )
    else:
        test_set = corrupt_data_tmp(
            args=args, orgin_set=SpeechCommandsV2(
                root_path=args.dataset_root_path, mode='testing', download=True,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
                ])
            )
        )

    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    if args.corruption_type == 'PSH':
        mlt_store_to(dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name, data_tfs=[DoNothing()])
    else:
        batch_store_to(
            data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8),
            root_path=dataset_root_path, index_file_name=index_file_name
        )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ReduceChannel()]
    )
    adpt_set = MultiTFDataset(
        dataset=mlt_load_from(root_path=dataset_root_path, index_file_name=index_file_name),
        tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                ReduceChannel()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                ReduceChannel()
            ])
        ]
    )
    return test_set, adpt_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
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
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'TTDA')
    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    test_set, adapt_set = sc_corruption_set(args)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )

    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weigth(args=args, hubert=hubert, clsf=clsf, mode='origin')
    optimizer = build_optimizer(lr=args.lr, auT=hubert, auC=clsf, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay)

    def inferecing(max_accu:float) -> tuple[float, float]:
        accuracy = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(adapt_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(
                hubert.state_dict(), 
                os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt')
            )
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt')
            )
        return accuracy, max_accu
    
    max_accu = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
            
        print('Inferencing...')
        accuracy, max_accu = inferecing(max_accu)

        print('Adaptating...')
        hubert.train(); clsf.train()
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.; ttl_gent_loss = 0.
        for fs1, fs2, _ in tqdm(adapt_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsf(hubert(fs1)[0])
            os2, _ = clsf(hubert(fs2)[0])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1) + entropy(args, os2)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os1, q=args.gent_q)

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
    print('Finalizing...')
    accuracy, max_accu = inferecing(max_accu)
    wandb_run.log(
        data={
            'Loss/ttl_loss': ttl_loss / ttl_size,
            'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
            'Loss/Entropy loss': ttl_ent_loss / ttl_size,
            'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
            'Adaptation/accuracy': accuracy,
            'Adaptation/LR': learning_rate,
            'Adaptation/max_accu': max_accu,
        }, step=args.max_epoch, commit=True
    )