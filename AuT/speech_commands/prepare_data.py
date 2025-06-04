import argparse
import os

import torch
from torch import nn
from torchaudio import transforms as a_transforms
from torch.utils.data import Dataset

from lib.utils import make_unless_exits, print_argparse
from lib.spdataset import SpeechCommandsV2
from lib.component import Components, AudioPadding, BackgroundNoiseByFunc, time_shift, DoNothing
from lib.dataset import MgDataset, store_to
from AuT.speech_commands.analysis import noise_source

class GpuMultiTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tfs:list[nn.Module]):
        super(GpuMultiTFDataset, self).__init__()
        assert tfs is not None, 'No support'
        self.dataset = dataset
        self.tfs = tfs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        item = item.to('cuda')
        ret = [item.clone() for _ in range(len(self.tfs))]
        for i, tf in enumerate(self.tfs):
            if tf is not None:
                ret[i] = tf(ret[i]).to('cpu')
        ret.append(label)
        return tuple(ret)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--vocalsound_path', type=str)
    ap.add_argument('--cochlscene_path', type=str)
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--index_file_name', type=str, default='metaInfo.csv')
    ap.add_argument('--corruption_level', type=float)
    ap.add_argument('--corruption_type', type=str, choices=['doing_the_dishes', 'exercise_bike', 'running_tap', 'VocalSound', 'CochlScene'])

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_path = os.path.join(args.output_path, args.dataset)
    make_unless_exits(args.output_path)

    print_argparse(args)
    ##########################################

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    corrupted_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='testing', download=True, 
        data_tf=Components(transforms=[
            AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
            BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True),
        ])
    )
    wk_str_set = GpuMultiTFDataset(
        dataset=corrupted_set,
        tfs=[
            DoNothing().to('cuda'),
            Components(transforms=[
                a_transforms.PitchShift(sample_rate=sample_rate, n_steps=4, n_fft=512),
                time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            ]).to('cuda')
        ]
    )
    store_to(
        dataset=MgDataset(dataset=wk_str_set), 
        root_path=os.path.join(args.output_path, args.corruption_type, str(args.corruption_level)), 
        index_file_name=args.index_file_name
    )