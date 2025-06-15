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
from lib.dataset import RandomChoiceSet, mlt_store_to, mlt_load_from, MultiTFDataset
from lib.component import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.component import BackgroundNoiseByFunc, DoNothing, time_shift
from lib.acousticDataset import CochlScene
from AuT.speech_commands.train import build_model
from AuT.lib.model import FCETransform, AudioClassifier

def merge_outs(o1:torch.Tensor, o2:torch.Tensor, o3:torch.Tensor, softmax:bool=False) -> torch.Tensor:
    if softmax:
        from torch.nn import functional as F
        return (F.softmax(o1, dim=1) + F.softmax(o2, dim=1) + F.softmax(o3, dim=1)) / 3.
    else: return (o1 + o2 + o3) / 3.

def clean_cache(cache_path) -> None:
    import shutil
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

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

def noise_source(args:argparse, source_type:str):
    def expanding(noise: torch.Tensor, target_length=16000) -> torch.Tensor:
        noise_len = noise.shape[1]
        i = 1
        while i * noise_len < target_length: i += 1
        if i == 1: return noise
        else: return noise.repeat([1, i])

    if source_type == 'VocalSound':
        noise_set = VocalSound(root_path=args.vocalsound_path, mode='train', include_rate=False, version='16k')
        source = RandomChoiceSet(dataset=noise_set)
        return lambda: expanding(source[0][0])
    elif source_type in ['doing_the_dishes', 'exercise_bike', 'running_tap']:
        noise_set = BackgroundNoiseDataset(root_path=args.background_path, include_rate=False)
        for noise, noise_type in noise_set:
            if noise_type == source_type:
                source = noise
                break
        return lambda: source
    elif source_type == 'CochlScene':
        noise_set = CochlScene(root_path=args.cochlscene_path, mode='train', include_rate=False)
        source = RandomChoiceSet(dataset=noise_set)
        return lambda: expanding(source[0][0])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--vocalsound_path', type=str)
    ap.add_argument('--cochlscene_path', type=str)
    ap.add_argument('--corruption_level', type=float)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--analysis_file', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)
    ap.add_argument('--adapted_auT_weight', type=str)
    ap.add_argument('--adapted_cls_weight', type=str)
    ap.add_argument('--noise_types', type=str)
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
    args.noise_types = [it.strip() for it in str(args.noise_types).split(',')]

    print_argparse(args)
    ###########################################################

    records = pd.DataFrame(columns=['dataset', 'module', 'param_num', 'corruption_type', 'corruption_level', 'adaptation', 'accuracy', 'error_rate'])

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
    records.loc[len(records)] = [args.dataset, arch, param_num, np.nan, np.nan, np.nan, accuracy, 100.-accuracy]

    print('Corrupted')
    for noise_type in args.noise_types:
        print(f'Process {noise_type}...')
        if args.dataset == 'SpeechCommandsV2':
            corrupted_set = SpeechCommandsV2(
                root_path=args.dataset_root_path, mode='testing', download=True,
                data_tf=Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                    BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=noise_type), is_random=True),
                ])
            )
        elif args.dataset == 'SpeechCommandsV1':
            corrupted_set = SpeechCommandsV1(
                root_path=args.dataset_root_path, mode='test', include_rate=False,
                data_tfs=Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                    BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=noise_type), is_random=True),
                ])
            )
        cache_path = os.path.join(args.cache_path, args.dataset, noise_type, str(args.corruption_level))
        index_file_name = 'metaInfo.csv'
        mlt_store_to(dataset=corrupted_set, root_path=cache_path, index_file_name=index_file_name, data_tfs=[DoNothing()])
        corrupted_set = mlt_load_from(
            root_path=cache_path, index_file_name=index_file_name,
            data_tfs=[
                Components(transforms=[
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
        corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)
        accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=corrupted_loader)
        print(f'Corrupted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, np.nan, accuracy, 100.-accuracy]

        print('Adapted')
        auT1, cls1 = build_model(args)
        load_weight(args, mode='adaption', auT=auT1, auC=cls1)
        accuracy = inference(args=args, auT=auT1, auC=cls1, data_loader=corrupted_loader)
        print(f'accuracy is: {accuracy:.4f}%, number of parameters is: {param_num}, sample size is: {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, 'CEG', accuracy, 100.-accuracy]

        corrupted_set = MultiTFDataset(
            dataset=mlt_load_from(root_path=cache_path, index_file_name=index_file_name),
            tfs=[
                Components(transforms=[
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 104
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
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

        corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        load_weight(args, mode='adaption', auT=auT1, auC=cls1)
        auT1.eval(); cls1.eval()
        ttl_corr = 0. 
        ttl_size = 0.
        for f1, f2, f3, labels in tqdm(corrupted_loader):
            f1, f2, f3, labels = f1.to(args.device), f2.to(args.device), f3.to(args.device), labels.to(args.device)

            with torch.no_grad():
                o1, _ = cls1(auT1(f1)[0])
                o2, _ = cls1(auT1(f2)[0])
                o3, _ = cls1(auT1(f3)[0])
                outputs = merge_outs(o1, o2, o3, softmax=args.softmax)
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
        accuracy = ttl_corr / ttl_size * 100.
        print(f'augment election accuracy is: {accuracy:.4f}%, number of parameters is: {param_num}, sample size is: {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, 'aug-elec', accuracy, 100.-accuracy]

        clean_cache(cache_path)

    records.to_csv(os.path.join(args.output_path, args.analysis_file))