import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.dataset import mlt_load_from, MultiTFDataset
from lib.component import Components, ReduceChannel, AudioClip
from HuBERT.ReefSet.train import build_model, inference
from HuBERT.ReefSet.analysis import load_weight

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Ablation_study')

    print_argparse(args)
    make_unless_exits(args.output_path)
    ##########################################
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    records = pd.DataFrame(columns=['dataset', 'algorithm', 'type', 'before-adaptation', 'after-adaptation'])

    noise_set = mlt_load_from(
        root_path=args.dataset_root_path, index_file_name='meta_info.csv', class_num=args.class_num,
    )

    test_loader = DataLoader(
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, 
        dataset=MultiTFDataset(
            dataset=noise_set, tfs=[
                Components(transforms=[
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                    ReduceChannel()
                ])
            ]
        )
    )

    print('Before adaptation analyzing...')
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    b_roc_auc = inference(args=args, hub=hubert, clsf=clsf, loader=test_loader)
    print(f'Before adaptation ROC-AUC is: {b_roc_auc:.4f}, sample size is: {len(noise_set)}')

    print('After adaptation analyzing...')
    hub_wght_pth = os.path.join(args.adpt_wght_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-fix.pt')
    hubert.load_state_dict(state_dict=torch.load(hub_wght_pth, weights_only=True))
    clsf_wght_pth = os.path.join(args.adpt_wght_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-fix.pt')
    clsf.load_state_dict(state_dict=torch.load(clsf_wght_pth, weights_only=True))
    a_roc_auc = inference(args=args, hub=hubert, clsf=clsf, loader=test_loader)
    print(f'After adaptation ROC-AUC is: {a_roc_auc:.4f}, sample size is: {len(noise_set)}')

    records.loc[len(records)] = [args.dataset, args.arch, 'Fix-corruption', b_roc_auc, a_roc_auc]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))
    print('END!')