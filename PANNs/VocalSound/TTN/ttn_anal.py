import argparse
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta, VocalSoundC
from lib.component import Components, ReduceChannel
from lib.normAdapt import NormAdapt
from PANNs.lib.utils import build_model, load_weight, TentPANNs
from PANNs.VocalSound.utils import inference

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    pan, clsf = build_model(args, use_pre_weight=False)
    param_no = count_ttl_params(pan) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        adpt_set = VocalSoundC(
            root_path=args.adpt_set_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            data_tf=Components(transforms=[
                Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
                ReduceChannel()
            ])
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.adpt_batch_size, shuffle=True, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )
        eval_set = VocalSoundC(
            root_path=args.eval_set_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            data_tf=Components(transforms=[
                Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
                ReduceChannel()
            ])
        )
        eval_loader = DataLoader(
            dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        pan, clsf = build_model(args, use_pre_weight=False)
        load_weight(args=args, panns=pan, clsf=clsf, mode='origin')
        orig_accu = inference(args=args, pan=pan, clsf=clsf, data_loader=eval_loader)

        print('TTN adapting...')
        tp_model = TentPANNs(pan=pan, clsf=clsf)
        ttn_model = NormAdapt(model=tp_model, momentum=args.lr_momentum, reset_states=False, no_states=False).to(device=args.device)
        for features, labels in tqdm(adpt_loader):
            features = features.to(args.device)
            ttn_model(features)
        print('Adaptation analyzing...')
        adpt_accu = inference(args=args, pan=pan, clsf=clsf, data_loader=eval_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy: {orig_accu:.4f}, adapted accuracy: {adpt_accu:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_accu, adpt_accu, adpt_accu - orig_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--adpt_batch_size', type=int, default=64)

    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--pan_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)

    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--orig_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * constants.pann_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'TTN')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2']) 