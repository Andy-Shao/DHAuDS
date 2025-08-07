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

from lib import constants
from lib.utils import print_argparse
from lib.corruption import corruption_meta, ReefSetC, corrupt_data
from lib.dataset import MultiTFDataset
from lib.component import Components, AudioPadding, AudioClip

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
        if cmeta.type == 'TST':
            corrupted_set = corrupt_data(
                orgin_set=ReefSet(
                    root_path=args.dataset_root_path, meta_file=meta_file
                ), corruption_type=cmeta.type, corruption_level=cmeta.level, 
                enq_path=args.enq_path, sample_rate=args.sample_rate, end_path=args.end_path,
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
                        AudioClip(max_length=max_length, mode='mid', is_random=False)
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
            corrupted_set = corrupt_data(
                orgin_set=test_set, corruption_level=cmeta.level, corruption_type=cmeta.type, enq_path=args.enq_path,
                sample_rate=args.sample_rate, end_path=args.end_path, ensc_path=args.ensc_path
            )
        if cmeta.type == 'PSH':
            store_to(dataset=corrupted_set, root_path=ops_path, sample_rate=args.sample_rate)
        else:
            batch_store_to(
                data_loader=DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False,
                    drop_last=False, num_workers=args.num_workers),
                root_path=ops_path, sample_rate=args.sample_rate
            )
    shutil.copyfile(src=meta_file, dst=os.path.join(args.output_path, 'test_annotations.csv'))

    print('Testing...')
    for cmet in corruption_metas:
        print(f'Test {cmet.type}-{cmet.level}')
        for feature, label in tqdm(ReefSetC(root_path=args.output_path, corruption_type=cmet.type, corruption_level=cmet.level)):
            pass