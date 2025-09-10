import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib.utils import make_unless_exits, print_argparse
from lib.spdataset import SpeechCommandsV2
from lib.component import Components, AudioPadding, AmplitudeToDB, FrequenceTokenTransformer
from AuT.SpeechCommandsV2.train import build_model, inference
from AuT.ReefSet.analysis import load_weight

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

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################

    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    test_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='testing', download=True, 
        data_tf=Components(transforms=[
            AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                mel_scale=mel_scale, n_mels=args.n_mels
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    aut, clsf = build_model(args=args)
    load_weight(args=args, aut=aut, clsf=clsf, mode='origin')
    accuracy = inference(args=args, aut=aut, clsf=clsf, data_loader=test_loader)
    print(f'Test Accuracy is: {accuracy:.4f}, sample size is: {len(test_set)}')