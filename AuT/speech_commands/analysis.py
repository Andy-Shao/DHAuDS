import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch 
import torchaudio.transforms as a_transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.spdataset import SpeechCommandsV2, VocalSound, BackgroundNoiseDataset, SpeechCommandsV1
from lib.dataset import RandomChoiceSet, MultiTFDataset
from lib.component import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.component import time_shift
from lib.acousticDataset import CochlScene
from AuT.speech_commands.train import build_model
from AuT.lib.model import FCETransform, AudioClassifier

def merge_outs(o1:torch.Tensor, o2:torch.Tensor, o3:torch.Tensor, softmax:bool=False) -> torch.Tensor:
    if softmax:
        from torch.nn import functional as F
        return (F.softmax(o1, dim=1) + F.softmax(o2, dim=1) + F.softmax(o3, dim=1)) / 3.
    else: return (o1 + o2 + o3) / 3.

def inference(args:argparse, auT:FCETransform, auC:AudioClassifier, data_loader:DataLoader) -> float:
    auT.eval()
    auC.eval()
    ttl_corr = 0.
    ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = auC(auT(features)[0])
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_size += labels.shape[0]
        ttl_corr += (preds == labels).sum().cpu().item()
    return ttl_corr / ttl_size * 100.

def load_weight(args:argparse, mode:str, auT:nn.Module, auC:nn.Module) -> None:
    if mode == 'origin':
        auT_weight_path = args.origin_auT_weight
        cls_weight_path = args.origin_cls_weight
    elif mode == 'adaption':
        auT_weight_path = args.adapted_auT_weight
        cls_weight_path = args.adapted_cls_weight
    else:
        raise Exception('No support')
    auT.load_state_dict(state_dict=torch.load(auT_weight_path, weights_only=True))
    auC.load_state_dict(state_dict=torch.load(cls_weight_path, weights_only=True))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--analysis_file', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)
    ap.add_argument('--adapted_auT_weight', type=str)
    ap.add_argument('--adapted_cls_weight', type=str)
    ap.add_argument('--softmax', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    elif args.dataset == 'SpeechCommandsV1':
        args.class_num = 30
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 'AuT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ###########################################################

    records = pd.DataFrame(columns=['dataset', 'module', 'param_num', 'adaptation', 'accuracy', 'error_rate'])

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    if args.dataset == 'SpeechCommandsV2':
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=True, 
            data_tf=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 104
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        )
    elif args.dataset == 'SpeechCommandsV1':
        test_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='test', include_rate=False,
            data_tfs=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 104
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    print('Original')
    auTmodel, clsmodel = build_model(args)
    load_weight(args=args, mode='origin', auT=auTmodel, auC=clsmodel)
    param_num = count_ttl_params(model=auTmodel) + count_ttl_params(model=clsmodel)
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Original testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(test_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, np.nan, accuracy, 100.-accuracy]

    print('Enhanced')
    load_weight(args=args, mode='adaption', auT=auTmodel, auC=clsmodel)
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Adapted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(test_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, 'TTA', accuracy, 100.-accuracy]

    print('Multi-Aug')
    if args.dataset == 'SpeechCommandsV1':
        aug_set = MultiTFDataset(
            dataset=SpeechCommandsV1(root_path=args.dataset_root_path, mode='test', include_rate=False),
            tfs=[
                Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 104
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                    time_shift(shift_limit=-.17, is_random=False, is_bidirection=False),
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 104
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                    time_shift(shift_limit=.17, is_random=False, is_bidirection=False),
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 104
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ])
            ]
        )
    aug_loader = DataLoader(dataset=aug_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    load_weight(args=args, mode='adaption', auT=auTmodel, auC=clsmodel)
    auTmodel.eval(); clsmodel.eval()
    ttl_corr, ttl_size = 0., 0.
    for f1, f2, f3, labels in tqdm(aug_loader):
        f1, f2, f3, labels = f1.to(args.device), f2.to(args.device), f3.to(args.device), labels.to(args.device)
        with torch.no_grad():
            o1, _ = clsmodel(auTmodel(f1)[0])
            o2, _ = clsmodel(auTmodel(f2)[0])
            o3, _ = clsmodel(auTmodel(f3)[0])
            outputs = merge_outs(o1, o2, o3, softmax=True)
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    accuracy = ttl_corr / ttl_size * 100.
    print(f'Adapted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(test_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, 'TTA+MltAug', accuracy, 100.-accuracy]

    records.to_csv(os.path.join(args.output_path, args.analysis_file))