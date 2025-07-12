import argparse
import os
import pandas as pd
import shutil
import time

import torch
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

from lib.utils import print_argparse, make_unless_exits, count_ttl_params
from lib.dataset import mlt_load_from, mlt_store_to
from lib.component import ASTFeatureExt, DoNothing
from lib import constants
from AST.speech_commands.ttda import corrupt_data, build_model, inference

def load_weight(args:argparse.Namespace, model:ASTForAudioClassification) -> None:
    model.load_state_dict(state_dict=torch.load(args.adpt_wght_pth, weights_only=True))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END', 'TST+PSH', 'DP+PSH'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--repeat_no', type=int, default=3)

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AST'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    records = pd.DataFrame(columns=['index', 'Dataset', 'Algorithm', 'Param No.', 'No-adapted', 'Adapted'])

    cpt_set = corrupt_data(args, mode='eval')
    for idx in range(args.repeat_no):

        fe, ast = build_model(args)
        param_no = count_ttl_params(model=ast)
        dataset_root_path = os.path.join(os.path.join(args.cache_path, args.dataset))
        index_file_name = 'metaInfo.csv'
        mlt_store_to(
            dataset=cpt_set, root_path=dataset_root_path, index_file_name=index_file_name,
            data_tfs=[DoNothing()]
        )
        test_set = mlt_load_from(
            root_path=dataset_root_path, index_file_name=index_file_name, 
            data_tfs=[ASTFeatureExt(feature_extractor=fe, sample_rate=args.sample_rate, mode='batch')]
        )
        test_loader = DataLoader(
            dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, 
            pin_memory=True
        )

        print('Non-adapted')
        no_adpt_accu = inference(args=args, ast=ast, data_loader=test_loader)
        print(f'Non-adapted accuracy: {no_adpt_accu:.4f}%, sample size: {len(test_set)}')
    
        print('Adapted')
        load_weight(args=args, model=ast)
        adpt_accu = inference(args=args, ast=ast, data_loader=test_loader)
        print(f'Adapted accuracy: {adpt_accu:.4f}%, sample size: {len(test_set)}')
        records.loc[len(records)] = [
            idx, constants.dataset_dic[args.dataset], constants.architecture_dic[args.arch], 
            param_no, no_adpt_accu, adpt_accu
        ]
    no_adpt_accu = records['No-adapted'].mean()
    adpt_accu = records['Adapted'].mean()
    records.loc[len(records)] = [
        'Mean', constants.dataset_dic[args.dataset], constants.architecture_dic[args.arch], 
        param_no, no_adpt_accu, adpt_accu
    ]
    print(f'No-adapted mean accuracy: {no_adpt_accu:.4f}%, adapted mean accuracy: {adpt_accu:.4f}%')

    records.to_csv(os.path.join(
        args.output_path, 
        f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.csv'
    ))