import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, Normalize

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta, ReefSetC
from lib.component import OneHot2Index
from CoNMix.lib.utils import ExpandChannel, Components, cal_norm
from CoNMix.SpeechCommandsV2.analysis import load_weight
from CoNMix.SpeechCommandsV2.train import load_models
from CoNMix.ReefSet.train import inference

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    n_mels=123
    hop_length=250
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    modelF, modelB, modelC = load_models(args)
    param_no = count_ttl_params(modelF) + count_ttl_params(modelB) + count_ttl_params(modelC)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        tf_array = [
            MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            AmplitudeToDB(top_db=80.),
            ExpandChannel(out_channel=3),
            Resize((224, 224), antialias=False)
        ]
        if args.normalized:
            print('calculating adapted mean and standard deviation')
            adpt_set = ReefSetC(
                root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
                data_tf=Components(transforms=tf_array)
            )
            adpt_mean, adpt_std = cal_norm(loader=DataLoader(dataset=adpt_set, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
            tf_array.append(Normalize(mean=adpt_mean, std=adpt_std))
        adpt_set = ReefSetC(
            root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            data_tf=Components(transforms=tf_array), label_mode='single', label_tf=OneHot2Index()
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        load_weight(args=args, mF=modelF, mB=modelB, mC=modelC, mode='origin')
        orig_roc_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=adpt_loader, device=args.device, class_num=args.class_num)
        print('Adaptation analyzing...')
        load_weight(args=args, mF=modelF, mB=modelB, mC=modelC, mode='adaptation', metaInfo=cmeta)
        adpt_roc_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=adpt_loader, device=args.device, class_num=args.class_num)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted roc-auc: {orig_roc_auc:.4f}, adapted roc-auc: {adpt_roc_auc:.4f}')
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
    ap.add_argument('--adpt_wght_path', type=str)

    # CoNMix
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])

    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
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