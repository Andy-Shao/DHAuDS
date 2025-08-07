import argparse
import os
import pandas as pd

import torch 
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta, UrbanSound8KC
from lib.component import Components, AudioClip, ReduceChannel
from HuBERT.UrbanSound8K.train import inference, build_model
from HuBERT.ReefSet.analysis import load_weight

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted'])
    corruption_metas = corruption_meta(corrupytion_types=corruption_types, corruption_levels=corruption_levels)
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        adpt_hubert, adpt_clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
        load_weight(args=args, hubert=adpt_hubert, clsf=adpt_clsf, mode='adaptation', metaInfo=cmeta)

        adpt_set = UrbanSound8KC(
            root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
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
        orig_f1 = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=adpt_loader)
        print('Adaptation analyzing...')
        adpt_f1 = inference(args=args, hubert=adpt_hubert, clsModel=adpt_clsf, data_loader=adpt_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted f1: {orig_f1:.4f}, adapted f1: {adpt_f1:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_f1, adpt_f1]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 16000
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN'], corruption_levels=['L2']) 