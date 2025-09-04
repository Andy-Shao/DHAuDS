import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.spdataset import VocalSound
from lib.component import Components, AudioClip, AudioPadding, AmplitudeToDB, FrequenceTokenTransformer
from lib.component import MelSpectrogramPadding
from AuT.VocalSound.train import build_model, inference
from AuT.ReefSet.analysis import load_weight

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

    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    test_set = VocalSound(
        root_path=args.dataset_root_path, mode='test', include_rate=False, version='16k', 
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, 
                n_mels=args.n_mels, mel_scale=mel_scale
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            MelSpectrogramPadding(target_length=args.target_length),
            FrequenceTokenTransformer()
        ])
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )

    aut, clsf = build_model(args)
    load_weight(args=args, aut=aut, clsf=clsf, mode='origin')

    accuracy = inference(args=args, aut=aut, clsf=clsf, data_loader=test_loader)
    print(f'Test accuracy is: {accuracy:.4f}, sample size is: {len(test_set)}')