import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta, UrbanSound8KC
from lib.component import Components, ReduceChannel, AudioClip
from PANNs.lib.utils import build_model, load_weight
from .utils import inference

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    pan, clsf = build_model(args, use_pre_weight=False)
    param_no = count_ttl_params(pan) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        adpt_set = UrbanSound8KC(
            root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type,
            data_tf=Components(transforms=[
                Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
                AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                ReduceChannel()
            ])
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        load_weight(args=args, panns=pan, clsf=clsf, mode='origin')
        orig_f1 = inference(args=args, pan=pan, clsf=clsf, data_loader=adpt_loader)
        print('Adaptation analyzing...')
        load_weight(args=args, panns=pan, clsf=clsf, mode='adaptation', metaInfo=cmeta)
        adpt_f1 = inference(args=args, pan=pan, clsf=clsf, data_loader=adpt_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy: {orig_f1:.4f}, adapted accuracy: {adpt_f1:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_f1, adpt_f1, adpt_f1 - orig_f1]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = constants.pann_sample_rate
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * constants.pann_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2']) 