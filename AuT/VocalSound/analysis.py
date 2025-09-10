import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib.utils import print_argparse, make_unless_exits, count_ttl_params
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer
from lib.component import MelSpectrogramPadding
from lib.corruption import corruption_meta, VocalSoundC
from AuT.VocalSound.train import build_model, inference
from AuT.ReefSet.analysis import load_weight

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    aut, clsf = build_model(args=args)
    load_weight(args=args, aut=aut, clsf=clsf, mode='origin')
    param_no = count_ttl_params(aut) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        adpt_aut, adpt_clsf = build_model(args=args)
        load_weight(args=args, aut=adpt_aut, clsf=adpt_clsf, mode='adaptation', metaInfo=cmeta)

        adpt_set = VocalSoundC(
            root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type,
            data_tf=Components(transforms=[
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    mel_scale=mel_scale, n_mels=args.n_mels
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Non-adaptation analyzing...')
        orig_f1 = inference(args=args, aut=aut, clsf=clsf, data_loader=adpt_loader)
        print('Adaptation analyzing...')
        adpt_f1 = inference(args=args, aut=adpt_aut, clsf=adpt_clsf, data_loader=adpt_loader)
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy: {orig_f1:.4f}, adapted accuracy: {adpt_f1:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', orig_f1, adpt_f1, adpt_f1 - orig_f1]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * args.sample_rate)
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