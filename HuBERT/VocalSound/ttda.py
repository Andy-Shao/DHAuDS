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

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib import constants
from lib.spdataset import VocalSound
from lib.acousticDataset import DEMAND, QUTNOISE
from lib.dataset import batch_store_to, mlt_load_from, MergSet, MultiTFDataset, GpuMultiTFDataset, mlt_store_to
from lib.spdataset import SpeechCommandsV2, SpeechCommandsBackgroundNoise
from lib.component import Components, AudioPadding, AudioClip, ReduceChannel, time_shift, Stereo2Mono, DoNothing
from lib.corruption import WHN, DynEN, DynPSH, DynTST
from lib.loss import entropy, g_entropy, nucnm
from lib.lr_utils import build_optimizer, lr_scheduler
from HuBERT.VocalSound.train import build_model, inference

def ensc_noises(args:argparse.Namespace, noise_modes:list[str]) -> list[torch.Tensor]:
    SpeechCommandsV2(root_path=args.noise_path, mode='testing', download=True)
    noise_set = SpeechCommandsBackgroundNoise(
        root_path=os.path.join(args.noise_path, 'speech_commands_v0.02', 'speech_commands_v0.02'), 
        include_rate=False
    )
    print('Loading noise files...')
    noises = []
    for noise, noise_type in tqdm(noise_set):
        if noise_type in noise_modes:
            noises.append(noise)
    print(f'TTL noise size is: {len(noises)}')
    return noises

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

def load_weigth(args:argparse.Namespace, hubert:nn.Module, clsf:nn.Module, mode:str='origin') -> None:
    if mode == 'origin':
        hub_pth = args.hub_wght_pth
        clsf_pth = args.clsf_wght_pth
    elif mode == 'adaption':
        hub_pth = args.adpt_hub_wght_pth
        clsf_pth = args.adpt_clsf_wght_pth
    hubert.load_state_dict(state_dict=torch.load(hub_pth, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(clsf_pth, weights_only=True))

def vs_corruption_data(args:argparse.Namespace) -> Dataset:
    if args.corruption_type == 'TST':
        test_set = corrupt_data(
            args=args, orgin_set=VocalSound(
                root_path=args.dataset_root_path, mode='test', include_rate=False, version='16k',
            )
        )
        test_set = MultiTFDataset(
            dataset=test_set, 
            tfs=[
                Components(transforms=[
                    AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=10*args.sample_rate, mode='head', is_random=False),
                ])
            ]
        )
    else:
        test_set = corrupt_data(
            args=args, orgin_set=VocalSound(
                root_path=args.dataset_root_path, mode='test', include_rate=False, version='16k',
                data_tf=Components(transforms=[
                    AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=10*args.sample_rate, mode='head', is_random=False),
                ])
            )
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    if args.corruption_type == 'PSH':
        mlt_store_to(
            dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name, 
            data_tfs=[DoNothing()]
        )
    else:
        batch_store_to(
            data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8),
            root_path=dataset_root_path, index_file_name=index_file_name, f_num=1
        )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ ReduceChannel() ]
    )
    adapt_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                ReduceChannel()
            ])
        ]
    )
    return test_set, adapt_set

def corrupt_data(args:argparse.Namespace, orgin_set:Dataset) -> Dataset:
    if args.corruption_level == 'L1':
        snrs = constants.DYN_SNR_L1
        n_steps = constants.DYN_PSH_L1
        rates = constants.DYN_TST_L1
    elif args.corruption_level == 'L2':
        snrs = constants.DYN_SNR_L2
        n_steps = constants.DYN_PSH_L2
        rates = constants.DYN_TST_L2
    if args.corruption_type == 'WHN':
        test_set = MultiTFDataset(dataset=orgin_set, tfs=[WHN(lsnr=snrs[0], rsnr=snrs[2], step=snrs[1])])
    elif args.corruption_type == 'ENQ':
        noise_modes = constants.ENQ_NOISE_LIST
        test_set = MultiTFDataset(dataset=orgin_set, tfs=[
            DynEN(noise_list=enq_noises(args=args, noise_modes=noise_modes), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
        ])
    elif args.corruption_type == 'END1':
        noise_modes = constants.END1_NOISE_LIST
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=end_noises(args=args, noise_modes=noise_modes), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif args.corruption_type == 'END2':
        noise_modes = constants.END2_NOISE_LIST
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=end_noises(args=args, noise_modes=noise_modes), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif args.corruption_type == 'ENSC':
        noise_modes = constants.ENSC_NOISE_LIST
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=ensc_noises(args=args, noise_modes=noise_modes), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif args.corruption_type == 'PSH':
        test_set = GpuMultiTFDataset(
            dataset=orgin_set, tfs=[
                DynPSH(sample_rate=args.sample_rate, min_steps=n_steps[0], max_steps=n_steps[1], is_bidirection=True)
            ]
        )
    elif args.corruption_type == 'TST':
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynTST(min_rate=rates[0], step=rates[1], max_rate=rates[2], is_bidirection=True)
            ]
        )
    else:
        raise Exception('No support')
    return test_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
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
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
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
        name=f'{constants.architecture_dic[args.arch]}-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    test_set, adapt_set = vs_corruption_data(args)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weigth(args=args, hubert=hubert, clsf=clsf)
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
        for fs1, _ in tqdm(adapt_loader):
            fs1 = fs1.to(args.device)

            optimizer.zero_grad()
            os1 = clsf(hubert(fs1)[0][:, :2, :])

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