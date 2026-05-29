import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse, make_unless_exits, count_ttl_params
from lib.corruption import ReefSetC, corruption_meta
from lib.component import Components, AudioPadding, ReduceChannel, OneHot2Index, AudioClip
from PANNs.lib.utils import build_model, inference, load_weight

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    pan, clsf = build_model(args, use_pre_weight=False)
    param_no = count_ttl_params(pan) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        adpt_set = ReefSetC(
        root_path=args.dataset_root_path, corruption_level=args.corruption_level, corruption_type=args.corruption_type,
        data_tf=Components(transforms=[
            Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
            AudioPadding(max_length=args.audio_length, sample_rate=constants.pann_sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ]),
        label_tf=OneHot2Index()
    )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        load_weight(args=args, panns=pan, clsf=clsf, mode='origin')
        orig_roc_auc = inference(args=args, pan=pan, clsf=clsf, data_loader=adpt_loader)
        print('Adaptation analyzing...')
        load_weight(args=args, panns=pan, clsf=clsf, mode='adaptation', metaInfo=cmeta)
        adpt_roc_auc = inference(args=args, pan=pan, clsf=clsf, data_loader=adpt_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy: {orig_roc_auc:.4f}, adapted accuracy: {adpt_roc_auc:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_roc_auc, adpt_roc_auc, adpt_roc_auc - orig_roc_auc]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2']) 