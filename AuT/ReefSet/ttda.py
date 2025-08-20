import argparse
import os
import numpy as np
import random
import wandb

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.acousticDataset import ReefSet
from lib.dataset import MultiTFDataset, mlt_load_from, mlt_store_to, batch_store_to
from lib.component import Components, AudioPadding, AudioClip, time_shift, OneHot2Index
from lib.component import AmplitudeToDB, FrequenceTokenTransformer, DoNothing
from AuT.ReefSet.train import build_model, inference

def load_weigth(args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode:str='origin') -> None:
    if mode == 'origin':
        aut_pth = args.aut_wght_pth
        clsf_pth = args.clsf_wght_pth
    elif mode == 'adaption':
        aut_pth = args.adpt_aut_wght_pth
        clsf_pth = args.adpt_clsf_wght_pth
    aut.load_state_dict(state_dict=torch.load(aut_pth, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(clsf_pth, weights_only=True))

def rs_corrupt_data(args:argparse.Namespace) -> tuple[Dataset, Dataset]:
    from lib.corruption import corrupt_data

    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195
    if args.corruption_type == 'TST':
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False, label_mode='single', label_tf=OneHot2Index()
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
        test_set = MultiTFDataset(
            dataset=test_set, tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ])
            ]
        )
    else: 
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ]), label_mode='single', label_tf=OneHot2Index()
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    if args.corruption_type == 'PSH':
        mlt_store_to(
            dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name, data_tfs=[DoNothing()],
            is_one_hot_label=False
        )
    else:
        batch_store_to(
            data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8), 
            root_path=dataset_root_path, index_file_name=index_file_name, f_num=1, is_one_hot_label=False
        )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ 
            Components(transforms=[
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, n_mels=args.n_mels, 
                    hop_length=hop_length, mel_scale=mel_scale
                ), # 80 x 195
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        ], is_one_hot_label=False, class_num=args.class_num
    )
    adapt_set = MultiTFDataset(
        dataset=mlt_load_from(
            root_path=dataset_root_path, index_file_name=index_file_name, is_one_hot_label=False, class_num=args.class_num
        ), tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, n_mels=args.n_mels,
                    hop_length=hop_length, mel_scale=mel_scale
                ), # 80 x 195
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, n_mels=args.n_mels,
                    hop_length=hop_length, mel_scale=mel_scale
                ), # 80 x 195
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        ]
    )
    return test_set, adapt_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
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
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--aut_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
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
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
        
    test_set, adapt_set = rs_corrupt_data(args=args)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        pin_memory_device=args.device, num_workers=args.num_workers
    )
    aut, clsf = build_model(args=args)
    load_weigth(args, aut=aut, clsf=clsf, mode='origin')

    def inferecing(max_roc_auc:float) -> tuple[float, float]:
        val_roc_auc = inference(args=args, aut=aut, clsf=clsf, data_loader=test_loader)
        print(f'ROC-AUC is: {val_roc_auc:.4f}, sample size is: {len(test_set)}')
        if val_roc_auc >= max_roc_auc:
            max_roc_auc = val_roc_auc
            torch.save(
                aut.state_dict(), 
                os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}.pt')
            )
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'clsModel-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}.pt')
            )
        return val_roc_auc, max_roc_auc
    
    max_roc_auc = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
            
        print('Inferencing...')
        val_roc_auc, max_roc_auc = inferecing(max_roc_auc)
        exit()