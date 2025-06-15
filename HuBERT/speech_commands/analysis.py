import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch 
from torch.utils.data import DataLoader
import torchaudio

from lib.utils import print_argparse, make_unless_exits, count_ttl_params
from lib.spdataset import SpeechCommandsV1, SpeechCommandsV2
from lib.component import Components, AudioPadding, ReduceChannel, BackgroundNoiseByFunc, DoNothing, time_shift
from lib.dataset import mlt_load_from, mlt_store_to, MultiTFDataset
from AuT.speech_commands.analysis import load_weight, noise_source, clean_cache, merge_outs
from AuT.lib.model import AudioClassifier
from HuBERT.speech_commands.train import build_model

def inference(args:argparse.Namespace, hubert:torchaudio.models.Wav2Vec2Model, clsmodel:AudioClassifier, data_loader: DataLoader) -> float:
    hubert.eval(); clsmodel.eval()
    ttl_corr = 0.; ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = clsmodel(hubert(features)[0])
            _, preds = torch.max(outputs, dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

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
    arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True
    args.noise_types = [it.strip() for it in str(args.noise_types).split(',')]

    print_argparse(args)
    ###########################################################

    records = pd.DataFrame(columns=['dataset', 'module', 'param_num', 'corruption_type', 'corruption_level', 'adaptation', 'accuracy', 'error_rate'])

    sample_rate = 16000
    max_length = sample_rate
    if args.dataset == 'SpeechCommandsV1':
        test_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='test', include_rate=False,
            data_tfs=Components(transforms=[
                AudioPadding(max_length=max_length, sample_rate=sample_rate, random_shift=False),
                ReduceChannel()
            ])
        )
    elif args.dataset == 'SpeechCommandsV2':
        test_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', 
            data_tf=Components(transforms=[
                AudioPadding(max_length=max_length, sample_rate=sample_rate),
                ReduceChannel()
            ])
        )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    print("Original")
    hubert, clsmodel = build_model(args=args, pre_weight=True)
    load_weight(args=args, mode='origin', auT=hubert, auC=clsmodel)
    param_num = count_ttl_params(model=hubert) + count_ttl_params(model=clsmodel)
    accuracy = inference(args=args, hubert=hubert, clsmodel=clsmodel, data_loader=test_loader)
    print(f'Original testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(test_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, np.nan, np.nan, np.nan, accuracy, 100.-accuracy]

    print('Corrupted')
    for noise_type in args.noise_types:
        print(f'Process {noise_type}...')
        if args.dataset == 'SpeechCommandsV1':
            corrupted_set = SpeechCommandsV1(
                root_path=args.dataset_root_path, mode='test', include_rate=False,
                data_tfs=Components(transforms=[
                    AudioPadding(max_length=max_length, sample_rate=sample_rate, random_shift=False),
                    BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=noise_type), is_random=True)
                ])
            )
        elif args.dataset == 'SpeechCommandsV2':
            SpeechCommandsV2(root_path=args.dataset_root_path, mode='testing', download=True)
            corrupted_set = SpeechCommandsV2(
                root_path=args.dataset_root_path, mode='testing', download=False,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=max_length, sample_rate=sample_rate, random_shift=False),
                    BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=noise_type), is_random=True)
                ])
            )
        cache_path = os.path.join(args.cache_path, args.dataset, noise_type, str(args.corruption_level))
        index_file_name = 'metaInfo.csv'
        mlt_store_to(dataset=corrupted_set, root_path=cache_path, index_file_name=index_file_name, data_tfs=[DoNothing()])
        corrupted_set = mlt_load_from(
            root_path=cache_path, index_file_name=index_file_name,
            data_tfs=[ReduceChannel()]
        )
        corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.num_workers)
        accuracy = inference(args=args, hubert=hubert, clsmodel=clsmodel, data_loader=corrupted_loader)
        print(f'Corrupted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, np.nan, accuracy, 100.-accuracy]

        print('Adapted')
        hubert, clsmodel = build_model(args=args, pre_weight=True)
        load_weight(args=args, mode='adaption', auT=hubert, auC=clsmodel)
        accuracy = inference(args=args, hubert=hubert, clsmodel=clsmodel, data_loader=corrupted_loader)
        print(f'accuracy is: {accuracy:.4f}%, number of parameters is: {param_num}, sample size is: {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, 'CEG', accuracy, 100.-accuracy]

        corrupted_set = MultiTFDataset(
            dataset=mlt_load_from(root_path=cache_path, index_file_name=index_file_name),
            tfs=[
                Components(transforms=[
                    ReduceChannel()
                ]),
                Components(transforms=[
                    time_shift(shift_limit=.1, is_random=False, is_bidirection=False),
                    ReduceChannel()
                ]),
                Components(transforms=[
                    time_shift(shift_limit=-.1, is_random=False, is_bidirection=False),
                    ReduceChannel()
                ])
            ]
        )
        corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        load_weight(args=args, mode='adaption', auT=hubert, auC=clsmodel)
        hubert.eval(); clsmodel.eval()
        ttl_corr=0.; ttl_size=0.
        for f1, f2, f3, labels in tqdm(corrupted_loader):
            f1, f2, f3, labels = f1.to(args.device), f2.to(args.device), f3.to(args.device), labels.to(args.device)

            with torch.no_grad():
                o1, _ = clsmodel(hubert(f1)[0])
                o2, _ = clsmodel(hubert(f2)[0])
                o3, _ = clsmodel(hubert(f3)[0])
                outputs = merge_outs(o1, o2, o3, softmax=args.softmax)
                _, preds = torch.max(outputs, dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
        accuracy = ttl_corr / ttl_size * 100.
        print(f'augment election accuracy is: {accuracy:.4f}%, number of parameters is: {param_num}, sample size is: {len(corrupted_set)}')
        records.loc[len(records)] = [args.dataset, arch, param_num, noise_type, args.corruption_level, 'aug-elec', accuracy, 100.-accuracy]

        clean_cache(cache_path)
    records.to_csv(os.path.join(args.output_path, args.analysis_file))