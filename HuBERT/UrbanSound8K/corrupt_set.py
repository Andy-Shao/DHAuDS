import argparse
import os
import shutil
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse
from lib.corruption import corruption_meta, corrupt_data, UrbanSound8KC
from lib.component import Components, AudioPadding, AudioClip, Stereo2Mono
from lib.dataset import MultiTFDataset

class UrbanSound8K(Dataset):
    def __init__(self, root_path:str, sample_rate:int, folds:list[int], data_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.sample_rate = sample_rate
        self.data_tf = data_tf
        self.data_list = UrbanSound8K.__cal_data_list__(root_path=root_path, folds=folds)

    @staticmethod
    def __cal_data_list__(root_path:str, folds:list[int]) -> pd.DataFrame:
        meta_file = os.path.join(root_path, 'metadata', 'UrbanSound8K.csv')
        meta_infos = pd.read_csv(meta_file, header=0)
        for idx, fold in enumerate(folds):
            meta_clip = meta_infos[meta_infos['fold'] == fold]
            if idx == 0:
                ret = meta_clip.copy(deep=True)
            else:
                ret = pd.concat([ret, meta_clip.copy(deep=True)], axis=0, ignore_index=True)
        return ret
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        meta_info = self.data_list.iloc[index]
        wavform, sample_rate = torchaudio.load(
            uri=os.path.join(self.root_path, 'audio', f'fold{meta_info['fold']}', meta_info['slice_file_name']),
            normalize=True
        )
        if sample_rate != self.sample_rate:
            resample_ops = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            wavform = resample_ops(wavform)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        return wavform, str(os.path.join(f'fold{meta_info['fold']}', meta_info['slice_file_name']))

def store_to(dataset:Dataset, root_path:str, sample_rate:int, data_tf:nn.Module=None) -> None:
    print(f'Store dataset into {root_path}')
    for feature, file_name in tqdm(dataset):
        if data_tf is not None:
            feature = data_tf(feature)
        torchaudio.save(
            uri=os.path.join(root_path, file_name), src=feature.detach(), sample_rate=sample_rate, encoding='PCM_S',
            bits_per_sample=16
        )

def batch_store_to(data_loader:DataLoader, root_path:str, sample_rate:int, data_tf:nn.Module=None) -> None:
    print(f'Store dataset into {root_path}')
    for fs, fns in tqdm(data_loader, total=len(data_loader)):
        for idx in range(len(fns)):
            feature = fs[idx]
            file_name = fns[idx]
            if data_tf is not None:
                feature = data_tf(feature)
            torchaudio.save(
                uri=os.path.join(root_path, file_name), src=feature, sample_rate=sample_rate, encoding='PCM_S',
                bits_per_sample=16
            )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--ensc_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 44100
        args.audio_length = int(4 * 44100)
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
    meta_file = os.path.join(args.output_path, 'metadata', 'UrbanSound8K.csv')
    corruption_metas = corruption_meta(
        corrupytion_types=['WHN', 'ENSC', 'PSH', 'TST'],
        corruption_levels=['L1', 'L2']
    )
    #makedirs
    for cmeta in corruption_metas:
        for fold in [8, 9, 10]:
            ops_path = os.path.join(args.output_path, 'audio', cmeta.type, cmeta.level, f'fold{fold}')
            os.makedirs(ops_path)
    os.makedirs(os.path.join(args.output_path, 'metadata'))
    meta_infos = UrbanSound8K.__cal_data_list__(root_path=args.dataset_root_path, folds=[8, 9, 10])
    meta_infos.to_csv(os.path.join(args.output_path, 'metadata', 'UrbanSound8K.csv'), header=True, index=False)
    print(f'Sample size is: {len(meta_infos)}')

    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, 'audio', cmeta.type, cmeta.level)
        if cmeta.type == 'TST':
            corrupted_set = corrupt_data(
                orgin_set=UrbanSound8K(
                    root_path=args.dataset_root_path, sample_rate=args.sample_rate, 
                    folds=[8, 9, 10], data_tf=Stereo2Mono()
                ), corruption_type=cmeta.type, corruption_level=cmeta.level, 
                enq_path=None, sample_rate=args.sample_rate, end_path=None,
                ensc_path=args.ensc_path
            )
            if cmeta.level == 'L1':
                max_length = int((1.+constants.DYN_TST_L1[2])*args.audio_length)
            elif cmeta.level == 'L2':
                max_length = int((1.+constants.DYN_TST_L2[2])*args.audio_length)
            else:
                raise Exception('No support')
            corrupted_set = MultiTFDataset(
                dataset=corrupted_set,
                tfs=[
                    Components(transforms=[
                        AudioPadding(max_length=max_length, sample_rate=args.sample_rate, random_shift=False),
                        AudioClip(max_length=max_length, mode='head', is_random=False)
                    ])
                ]
            )
        else:
            test_set = UrbanSound8K(
                root_path=args.dataset_root_path, sample_rate=args.sample_rate, folds=[8, 9, 10],
                data_tf=Components(transforms=[
                    Stereo2Mono(),
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ])
            )
            corrupted_set = corrupt_data(
                orgin_set=test_set, corruption_level=cmeta.level, corruption_type=cmeta.type, enq_path=None,
                sample_rate=args.sample_rate, end_path=None, ensc_path=args.ensc_path
            )
        if cmeta.type == 'PSH':
            store_to(dataset=corrupted_set, root_path=ops_path, sample_rate=args.sample_rate)
        else:
            batch_store_to(
                data_loader=DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False,
                    drop_last=False, num_workers=args.num_workers),
                root_path=ops_path, sample_rate=args.sample_rate,
            )

    print('Testing...')
    for cmet in corruption_metas:
        print(f'Test {cmet.type}-{cmet.level}')
        for feature, label in tqdm(UrbanSound8KC(root_path=args.output_path, corruption_type=cmet.type, corruption_level=cmet.level)):
            pass