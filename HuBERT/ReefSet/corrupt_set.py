import argparse
import os
import numpy as np
import random
import pandas as pd
import shutil
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse
from lib.corruption import corruption_meta, WHN, DynEN, DynPSH, DynTST, ReefSetC
from lib.spdataset import SpeechCommandsV2, SpeechCommandsBackgroundNoise
from lib.acousticDataset import DEMAND, QUTNOISE
from lib.dataset import MergSet, MultiTFDataset, GpuMultiTFDataset
from lib.component import Components, Stereo2Mono, AudioPadding

def store_to(dataset:Dataset, root_path:str, sample_rate:int, data_tf:nn.Module=None) -> None:
    print(f'Store dataset into {root_path}')
    for feature, file_name in tqdm(dataset):
        if data_tf is not None:
            feature = data_tf(feature)
        torchaudio.save(uri=os.path.join(root_path, file_name), src=feature, sample_rate=sample_rate)

def batch_store_to(data_loader:DataLoader, root_path:str, sample_rate:int, data_tf:nn.Module=None) -> None:
    print(f'Store dataset into {root_path}')
    for fs, fns in tqdm(data_loader, total=len(data_loader)):
        for idx in range(len(fns)):
            feature = fs[idx]
            file_name = fns[idx]
            if data_tf is not None:
                feature = data_tf(feature)
            torchaudio.save(uri=os.path.join(root_path, file_name), src=feature, sample_rate=sample_rate)

class ReefSet(Dataset):
    def __init__(self, root_path:str, meta_file:str, data_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.meta_info = pd.read_csv(meta_file, header=0)
        self.data_tf = data_tf
    
    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, index):
        meta = self.meta_info.iloc[index]
        wavform, sample_rate = torchaudio.load(os.path.join(self.root_path, 'full_dataset', meta['file_name']), normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        return wavform, meta['file_name']

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

def ensc_noises(args:argparse.Namespace, noise_modes:list[str]) -> list[torch.Tensor]:
    SpeechCommandsV2(root_path=args.ensc_path, mode='testing', download=True)
    noise_set = SpeechCommandsBackgroundNoise(
        root_path=os.path.join(args.ensc_path, 'speech_commands_v0.02', 'speech_commands_v0.02'), 
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
    demand_set = MergSet([DEMAND(root_path=args.end_path, mode=md, include_rate=False) for md in noise_modes])
    for wavform in tqdm(demand_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def enq_noises(args:argparse.Namespace, noise_modes:list[str] = ['CAFE', 'HOME', 'STREET']) -> list[torch.Tensor]:
    background_path = args.enq_path
    noises = []
    print('Loading noise files...')
    qutnoise_set = MergSet([
        QUTNOISE(
            root_path=background_path, mode=md, include_rate=False,
            data_tf=Components(transforms=[
                Resample(orig_freq=48000, new_freq=args.sample_rate),
                Stereo2Mono()
            ])
        ) for md in noise_modes
    ])
    for wavform in tqdm(qutnoise_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--enq_path', type=str)
    ap.add_argument('--end_path', type=str)
    ap.add_argument('--ensc_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    meta_file = './data/ReefSet/test_annotations.csv'
    corruption_metas = corruption_meta(
        corrupytion_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'],
        corruption_levels=['L1', 'L2']
    )

    #makedirs
    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, cmeta.type, cmeta.level)
        os.makedirs(ops_path)

    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, cmeta.type, cmeta.level)
        args.corruption_type = cmeta.type
        args.corruption_level = cmeta.level
        if cmeta.type == 'TST':
            corrupted_set = corrupt_data(args=args, orgin_set=ReefSet(
                root_path=args.dataset_root_path, meta_file=meta_file
            ))
            corrupted_set = MultiTFDataset(
                dataset=corrupted_set,
                tfs=[
                    Components(transforms=[
                        AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False)
                    ])
                ]
            )
        else:
            test_set = ReefSet(
                root_path=args.dataset_root_path, meta_file=meta_file, 
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False)
                ])
            )
            corrupted_set = corrupt_data(args=args, orgin_set=test_set)
        store_to(dataset=corrupted_set, root_path=ops_path, sample_rate=args.sample_rate)
    shutil.copyfile(src=meta_file, dst=os.path.join(args.output_path, 'test_annotations.csv'))

    print('Testing...')
    for feature, label in tqdm(ReefSetC(root_path=args.output_path, corruption_type='WHN', corruption_level='L1')):
        pass