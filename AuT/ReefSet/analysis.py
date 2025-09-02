import argparse
import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.corruption import CorruptionMeta, corruption_meta, ReefSetC
from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from AuT.ReefSet.train import build_model, inference
from lib.component import Components, AudioClip, AmplitudeToDB, FrequenceTokenTransformer

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        adpt_hubert, adpt_clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
        load_weight(args=args, hubert=adpt_hubert, clsf=adpt_clsf, mode='adaptation', metaInfo=cmeta)

        adpt_set = ReefSetC(
            root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            label_mode='single', data_tf=Components(transforms=[
                AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        orig_roc_auc = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=adpt_loader)
        print('Adaptation analyzing...')
        adpt_roc_auc = inference(args=args, hubert=adpt_hubert, clsModel=adpt_clsf, data_loader=adpt_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted roc-auc: {orig_roc_auc:.4f}, adapted roc-auc: {adpt_roc_auc:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_roc_auc, adpt_roc_auc, adpt_roc_auc - orig_roc_auc]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

def load_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None
) -> None:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        a_p = os.path.join(args.orig_wght_pth, f'aut-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(args.orig_wght_pth, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        a_p = os.path.join(args.adpt_wght_path, f'aut-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(args.adpt_wght_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

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

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 44100
        args.audio_length = int(4 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2']) 