import argparse
import numpy as np
import random
import os
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import add_noise

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.spdataset import SpeechCommandsV2
from lib.acousticDataset import DEMAND
from lib.dataset import mlt_store_to, mlt_load_from, MultiTFDataset
from lib.component import Components, AudioPadding, ReduceChannel, time_shift
from lib.lr_utils import build_optimizer, lr_scheduler
from lib.loss import nucnm, entropy, g_entropy, mse

from HuBERT.SpeechCommandsV2.train import build_model, inference
from HuBERT.ReefSet.analysis import load_weight

def build_noise_set(args:argparse.Namespace) -> Dataset:
    index_file_name='meta_info.csv'
    set_path = os.path.join(args.output_path, 'fix_clip_set')
    noise = DEMAND(root_path=args.background_path, mode='PSTATION', include_rate=False)[0]
    sc2 = SpeechCommandsV2(root_path=args.dataset_root_path, mode='testing', download=True,)
    mlt_store_to(
        dataset=sc2, root_path=set_path, index_file_name=index_file_name,
        data_tfs=[AddNoise(noise=noise, snr=5)]
    )
    return mlt_load_from(root_path=set_path, index_file_name=index_file_name, class_num=args.class_num)

class AddNoise(nn.Module):
    def __init__(self, noise:torch.Tensor, snr:float):
        super().__init__()
        self.noise = noise
        self.snr = snr

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return add_noise(waveform=x, noise=self.noise[:, 0:x.shape[1]], snr=torch.tensor([self.snr]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--mse_rate', type=float, default=0.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    args.arch = 'HuBERT'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    make_unless_exits(args.output_path)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}-fix', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    print('Create noise set')
    noise_set = build_noise_set(args)

    test_loader = DataLoader(
        dataset=MultiTFDataset(dataset=noise_set, tfs=[
            Components(transforms=[
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                ReduceChannel()
            ])
        ]), 
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    adapt_loader = DataLoader(
        dataset=MultiTFDataset(
            dataset=noise_set, tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                    time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                    ReduceChannel()
                ]),
                Components(transforms=[
                    AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                    time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                    ReduceChannel()
                ])
            ]
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    
    optimizer = build_optimizer(lr=args.lr, auT=hubert, auC=clsf, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay)

    def inferecing(max_accu:float) -> tuple[float, float]:
        accuracy = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}, sample size is: {len(noise_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(
                hubert.state_dict(), 
                os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-fix.pt')
            )
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-fix.pt')
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
        ttl_ent_loss = 0.; ttl_gent_loss = 0.; ttl_const_loss = 0.
        for fs1, fs2, _ in tqdm(adapt_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsf(hubert(fs1)[0])
            os2, _ = clsf(hubert(fs2)[0])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1) + entropy(args, os2)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os1, q=args.gent_q)
            const_loss = mse(args=args, out1=os1, out2=os2)

            loss = nucnm_loss + ent_loss + gent_loss + const_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()
            ttl_const_loss += const_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, 
                threshold=args.lr_threshold, momentum=args.lr_momentum
            )

        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/Entropy loss': ttl_ent_loss / ttl_size,
                'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
                'Loss/Consistency loss': ttl_const_loss / ttl_size,
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
            'Loss/Consistency loss': ttl_const_loss / ttl_size,
            'Adaptation/accuracy': accuracy,
            'Adaptation/LR': learning_rate,
            'Adaptation/max_accu': max_accu,
        }, step=args.max_epoch, commit=True
    )
    print('END!')