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

from lib import constants
from lib.utils import print_argparse
from lib.spdataset import SpeechCommandsV2 as SC2
from lib.corruption import corruption_meta, corrupt_data, DynTST, SpeechCommandsV2C
from lib.component import Components, AudioPadding

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

class SpeechCommandsV2(Dataset):
    meta_file = os.path.join('testing_list.txt')
    def __init__(self, root_path:str, data_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.data_tf = data_tf
        self.data_list = self.__cal_data_list_()

    def __cal_data_list_(self):
        with open(os.path.join(self.root_path, SpeechCommandsV2.meta_file), 'r') as f:
            data_list = f.readlines()
        return [it.strip() for it in data_list]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_path = self.data_list[index]
        wavform, sample_rate = torchaudio.load(os.path.join(self.root_path, file_path), normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        return wavform, file_path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
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
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
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
    SC2(root_path=args.dataset_root_path, mode='testing', download=True)
    dataset_root_path = os.path.join(args.dataset_root_path, 'speech_commands_v0.02', 'speech_commands_v0.02')
    corruption_metas = corruption_meta(
        corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'],
        corruption_levels=['L1', 'L2']
    )

    #makedirs
    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, cmeta.type, cmeta.level)
        for k,v in SC2.label_dict.items():
            os.makedirs(os.path.join(ops_path, k))

    for cmeta in corruption_metas:
        ops_path = os.path.join(args.output_path, cmeta.type, cmeta.level)
        if cmeta.type == 'TST':
            if cmeta.level == 'L1':
                rates = constants.DYN_TST_L1
            elif cmeta.level == 'L2':
                rates = constants.DYN_TST_L2
            test_set = SpeechCommandsV2(
                root_path=dataset_root_path, 
                data_tf=Components(transforms=[
                    DynTST(min_rate=rates[0], step=rates[1], max_rate=rates[2], is_bidirection=False),
                    AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
                ])
            )
        else:
            test_set = corrupt_data(
                orgin_set=SpeechCommandsV2(
                    root_path=dataset_root_path,
                    data_tf=AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
                ),
                corruption_level=cmeta.level, corruption_type=cmeta.type, enq_path=args.enq_path, sample_rate=args.sample_rate,
                end_path=args.end_path, ensc_path=args.ensc_path
            )

        if cmeta.type == 'PSH':
            store_to(dataset=test_set, root_path=ops_path, sample_rate=args.sample_rate)
        else:
            batch_store_to(
                data_loader=DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
                    num_workers=args.num_workers),
                root_path=ops_path, sample_rate=args.sample_rate
            )
    
    shutil.copy(src=os.path.join(dataset_root_path, SpeechCommandsV2.meta_file), dst=os.path.join(args.output_path, SpeechCommandsV2.meta_file))

    print('Testing...')
    for cmet in corruption_metas:
        print(f'Test {cmet.type}-{cmet.level}')
        for feature, label in tqdm(SpeechCommandsV2C(root_path=args.output_path, corruption_type=cmet.type, corruption_level=cmet.level)):
            pass