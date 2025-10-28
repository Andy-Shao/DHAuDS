import argparse
import numpy as np
import random
import wandb
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, RandomCrop, Normalize

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.spdataset import SpeechCommandsV2
from lib.dataset import batch_store_to, mlt_load_from, mlt_store_to, GpuMultiTFDataset
from lib.component import AudioPadding
from CoNMix.lib.utils import time_shift, Components, ExpandChannel, cal_norm

def sc_corruption_set(args:argparse.Namespace) -> tuple[Dataset, Dataset]:
    from lib.corruption import corrupt_data as corrupt_data_tmp, DynTST, DynPSH

    # max_ms = 1000
    n_mels=81
    hop_length=200
    if args.corruption_type == 'TST':
        if args.corruption_level == 'L1':
            rates = constants.DYN_TST_L1
        elif args.corruption_level == 'L2':
            rates = constants.DYN_TST_L2
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True,
            data_tf=Components(transforms=[
                DynTST(min_rate=rates[0], step=rates[1], max_rate=rates[2], is_bidirection=False),
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
                # pad_trunc(max_ms=max_ms, sample_rate=args.sample_rate)
            ])
        )
    else:
        test_set = corrupt_data_tmp(
            orgin_set=SpeechCommandsV2(
                root_path=args.dataset_root_path, mode='testing', download=True,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False)
                    # pad_trunc(max_ms=max_ms, sample_rate=args.sample_rate)
                ])
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
    
    # origin and weak set
    org_ds_path = os.path.join(args.cache_path, args.dataset, 'origin')
    index_file_name = 'metaInfo.csv'
    batch_store_to(
        data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8),
        root_path=org_ds_path, index_file_name=index_file_name
    )
    org_tf = [
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize((224, 224), antialias=False)
    ]
    if args.normalized:
        print('calculating original mean and standard deviation')
        org_set = mlt_load_from(
            root_path=org_ds_path, index_file_name=index_file_name, data_tfs=[Components(transforms=org_tf)]
        )
        org_mean, org_std = cal_norm(loader=DataLoader(dataset=org_set, batch_size=256, shuffle=False, drop_last=False))
        org_tf.append(Normalize(mean=org_mean, std=org_std))
    org_set = mlt_load_from(
        root_path=org_ds_path, index_file_name=index_file_name, data_tfs=[Components(transforms=org_tf)]
    )

    # strong augmentation set
    str_ds_path = os.path.join(args.cache_path, args.dataset, 'strong')
    str_set = GpuMultiTFDataset(
        dataset=mlt_load_from(
            root_path=org_ds_path, index_file_name=index_file_name
        ), 
        tfs=[Components(transforms=[
            # PitchShift(sample_rate=args.sample_rate, n_steps=4, n_fft=512),
            DynPSH(sample_rate=args.sample_rate, min_steps=4, max_steps=4, is_bidirection=False)
        ])], maintain_cpu=True
    )
    mlt_store_to(
        dataset=str_set, root_path=str_ds_path, index_file_name=index_file_name,
        data_tfs=[
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True)
        ]
    )
    str_tf = [
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize((256, 256), antialias=False),
        RandomCrop(224)
    ]
    if args.normalized:
        print('calculating strong augmentation mean and standard deviation')
        str_set = mlt_load_from(
            root_path=str_ds_path, index_file_name=index_file_name, 
            data_tfs=[Components(transforms=str_tf)]
        )
        str_mean, str_std = cal_norm(loader=DataLoader(dataset=str_set, batch_size=256, shuffle=False, drop_last=False))
        str_tf.append(Normalize(mean=str_mean, std=str_std))
    str_set = mlt_load_from(
        root_path=str_ds_path, index_file_name=index_file_name, 
        data_tfs=[Components(transforms=str_tf)]
    )

    return org_set, str_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    # CoNMix
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--cls_mode', type=str, default='soft_ce', choices=['logsoft_ce', 'soft_ce', 'logsoft_nll'])
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--ent_par', type=float, default=1.3)
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')
    ap.add_argument('--ent', action='store_true')
    ap.add_argument('--gent', action='store_true')

    ap.add_argument('--lr_decay1', type=float, default=0.1)
    ap.add_argument('--lr_decay2', type=float, default=1.0)
    ap.add_argument('--lr_gamma', type=int, default=30)

    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    ap.add_argument('--sdlr', type=int, default=1, help='lr_scheduler capable')
    ap.add_argument('--initc_num', type=int, default=1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--backup_weight', type=int, default=0)

    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'CoNMix'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'STDA')
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
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    org_set, str_set = sc_corruption_set(args=args)