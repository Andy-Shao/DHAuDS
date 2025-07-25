import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib import constants
from HuBERT.VocalSound.ttda import vs_corruption_data, load_weigth

def inferencing(args:argparse.Namespace, data_loader:DataLoader, mode:str='origin') -> tuple[float, int]:
    from HuBERT.VocalSound.train import inference, build_model
    assert mode in ['origin', 'adaption'], 'No support'
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weigth(args=args, hubert=hubert, clsf=clsf, mode=mode)
    accuracy = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=data_loader)
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)
    return accuracy, param_no

def analyze(inferencing:function, prepare_data:function, args:argparse.Namespace) -> pd.DataFrame:
    records = pd.DataFrame(columns=['idx', 'Dataset', 'Alg.', 'Param No.', 'Crpt-type', 'Crpt-level', 'No-adpt-accu', 'Adpt-accu'])

    for idx in range(args.repeat_time):
        data_loader = prepare_data(args)
        print('No-adapted analyzing...')
        no_adpt_accu, param_no = inferencing(args, mode='origin', data_loader=data_loader)
        print(f'No-adapted accuracy is: {no_adpt_accu:.4f}%, param no. is: {param_no}')
        print('Adapted analyzing...')
        adpt_accu, _ = inferencing(args, mode='adaption', data_loader=data_loader)
        print(f'Adapted accuracy is: {adpt_accu:.4f}%, param no is: {param_no}')
        records.loc[len(records)] = [idx, args.dataset, args.arch, param_no, args.corruption_type, args.corruption_level, no_adpt_accu, adpt_accu]
    return records

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)
    ap.add_argument('--adpt_hub_wght_pth', type=str)
    ap.add_argument('--adpt_clsf_wght_pth', type=str)
    ap.add_argument('--repeat_time', type=int, default=1)

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    records = analyze(inferencing=inferencing, prepare_data=vs_corruption_data, args=args)
    records.to_csv(os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}.csv'))