import argparse
import os
import shutil
import numpy as np
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from lib.utils import print_argparse
from lib.spdataset import VocalSound as VS
from lib.corruption import corruption_meta, corrupt_data
from lib.dataset import MultiTFDataset
from lib.component import Components, AudioPadding, AudioClip

class VocalSound(Dataset):
    def __init__(self, root_path:str, data_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        vs = VS(root_path=root_path, mode='test', version='16k')
        self.sample_list = vs.sample_list
        self.data_tf = data_tf

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        audioMeta = self.sample_list[index]
        wavform, sample_rate = torchaudio.load(audioMeta.file_path, normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        return wavform, str(audioMeta.file_path).split('/')[-1]

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
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
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
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
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
    corruption_metas = corruption_meta(
        corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'],
        corruption_levels=['L1', 'L2']
    )

    #makedirs
    data_file_path = os.path.join(args.output_path, 'datafiles')
    os.makedirs(data_file_path)
    shutil.copy(
        src=os.path.join(args.dataset_root_path, 'datafiles', 'te.json'), 
        dst=os.path.join(data_file_path, 'te.json')
    )
    meta_file_path = os.path.join(args.output_path, 'meta')
    os.makedirs(meta_file_path)
    shutil.copy(
        src=os.path.join(args.dataset_root_path, 'meta', 'te_meta.csv'),
        dst=os.path.join(meta_file_path, 'te_meta.csv')
    )
    shutil.copy(
        src=os.path.join(args.dataset_root_path, 'class_labels_indices_vs.csv'),
        dst=os.path.join(args.output_path, 'class_labels_indices_vs.csv')
    )
    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, 'audio_16k', cmeta.type, cmeta.level)
        os.makedirs(ops_path)

    # Generate data
    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, 'audio_16k', cmeta.type, cmeta.level)
        if cmeta.type == 'TST':
            test_set = corrupt_data(
                orgin_set=VocalSound(root_path=args.dataset_root_path), corruption_level=cmeta.level,
                corruption_type=cmeta.type, enq_path=args.enq_path, sample_rate=args.sample_rate,
                end_path=args.end_path, ensc_path=args.ensc_path
            )
            test_set = MultiTFDataset(
                dataset=test_set,
                tfs=[
                    Components(transforms=[
                        AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                        AudioClip(max_length=10*args.sample_rate, mode='head', is_random=False)
                    ])
                ]
            )
        else:
            test_set = corrupt_data(
                orgin_set=VocalSound(
                    root_path=args.dataset_root_path, 
                    data_tf=Components(transforms=[
                        AudioPadding(max_length=10*args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                        AudioClip(max_length=10*args.sample_rate, mode='head', is_random=False)
                    ])
                ), corruption_level=cmeta.level, corruption_type=cmeta.type, sample_rate=args.sample_rate,
                enq_path=args.enq_path, end_path=args.end_path, ensc_path=args.ensc_path
            )

        if cmeta.type == 'PSH':
            store_to(dataset=test_set, root_path=ops_path, sample_rate=args.sample_rate)
        else:
            batch_store_to(
                data_loader=DataLoader(
                    dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                    num_workers=args.num_workers
                ), root_path=ops_path, sample_rate=args.sample_rate
            )