import argparse
import os
import pandas as pd

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, Normalize

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta, CorruptionMeta, SpeechCommandsV2C
from CoNMix.SpeechCommandsV2.train import load_models
from CoNMix.SpeechCommandsV2.STDA import inference
from CoNMix.lib.utils import Components, ExpandChannel, cal_norm

def load_weight(
    args:argparse.Namespace, mF:nn.Module, mB:nn.Module, mC:nn.Module, mode='origin', metaInfo:CorruptionMeta=None
) -> None:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        F_path = os.path.join(args.orig_wght_pth, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelF.pt')
        B_path = os.path.join(args.orig_wght_pth, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelB.pt')
        C_path = os.path.join(args.orig_wght_pth, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-modelC.pt')
    elif mode == 'adaptation':
        F_path = os.path.join(args.adpt_wght_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}-modelF.pt')
        B_path = os.path.join(args.adpt_wght_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}-modelB.pt')
        C_path = os.path.join(args.adpt_wght_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}-modelC.pt')
    mF.load_state_dict(state_dict=torch.load(F_path, weights_only=True))
    mB.load_state_dict(state_dict=torch.load(B_path, weights_only=True))
    mC.load_state_dict(state_dict=torch.load(C_path, weights_only=True))

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    n_mels=81
    hop_length=200
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    modelF, modelB, modelC = load_models(args)
    param_no = count_ttl_params(modelF) + count_ttl_params(modelB) + count_ttl_params(modelC)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        tf_array = [
            MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, hop_length=hop_length, n_mels=n_mels),
            AmplitudeToDB(top_db=80.),
            ExpandChannel(out_channel=3),
            Resize((224, 224), antialias=False)
        ]
        if args.normalized:
            print('calculating adapted mean and standard deviation')
            adpt_set = SpeechCommandsV2C(
                root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
                data_tf=Components(transforms=tf_array)
            )
            adpt_mean, adpt_std = cal_norm(loader=DataLoader(dataset=adpt_set, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
            tf_array.append(Normalize(mean=adpt_mean, std=adpt_std))
        adpt_set = SpeechCommandsV2C(
            root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            data_tf=Components(transforms=tf_array)
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        load_weight(args=args, mF=modelF, mB=modelB, mC=modelC, mode='origin')
        orig_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=adpt_loader, device=args.device)
        print('Adaptation analyzing...')
        load_weight(args=args, mF=modelF, mB=modelB, mC=modelC, mode='adaptation', metaInfo=cmeta)
        adpt_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=adpt_loader, device=args.device)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy: {orig_auc:.4f}, adapted accuray: {adpt_auc:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_auc, adpt_auc, adpt_auc - orig_auc]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)

    # CoNMix
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])

    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'CoNMix'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'TST'], corruption_levels=['L1', 'L2']) 