import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
from torch.utils.data import DataLoader
import torchaudio

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.spdataset import SpeechCommandsV2
from lib.component import Components, AudioPadding, ReduceChannel, BackgroundNoiseByFunc
from HuBERT.speech_commands.train import build_model
from AuT.speech_commands.analysis import noise_source
from AuT.lib.model import AudioClassifier

def inference(args:argparse, auT:torchaudio.models.Wav2Vec2Model, auC:AudioClassifier, data_loader:DataLoader) -> float:
    auT.eval()
    auC.eval()
    ttl_corr = 0.
    ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = auC(auT(features)[0])
            _, preds = torch.max(outputs, dim=1)
        ttl_size += labels.shape[0]
        ttl_corr += (preds == labels).sum().cpu().item()
    ttl_accu = ttl_corr / ttl_size * 100.
    return ttl_accu

def load_weight(args:argparse, mode:str, auT:torchaudio.models.Wav2Vec2Model, auC:AudioClassifier) -> None:
    if mode == 'origin':
        auT_weight_path = args.origin_auT_weight
        cls_weight_path = args.origin_cls_weight
    auT.load_state_dict(state_dict=torch.load(auT_weight_path, weights_only=True))
    auC.load_state_dict(state_dict=torch.load(cls_weight_path, weights_only=True))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--background_type', default='VocalSound', choices=['VocalSound', 'SCV2-BG', 'CochlScene'])
    ap.add_argument('--corruption_level', type=float)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ###########################################################
    records = pd.DataFrame(columns=['dataset', 'module', 'param_num', 'corruption_type', 'corruption_level', 'accuracy', 'error_rate'])

    print('Original')
    test_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='testing', download=True,
        data_tf=Components(transforms=[
            AudioPadding(sample_rate=args.sample_rate, max_length=args.sample_rate, random_shift=False),
            ReduceChannel()
        ])
    )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    hubert, clsModel = build_model(args=args)
    param_num = count_ttl_params(model=hubert) + count_ttl_params(model=clsModel)
    load_weight(args=args, mode='origin', auT=hubert, auC=clsModel)
    accuracy = inference(args=args, auT=hubert, auC=clsModel, data_loader=test_loader)
    print(f'Original test set: accuracy is {accuracy:.4f}%, sample size is {len(test_set)}, number of params is {param_num}')
    records.loc[len(records)] = [args.dataset, arch, param_num, args.background_type, args.corruption_level, accuracy, 100.-accuracy]

    print('Corrupted')
    corrupted_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='testing', download=True,
        data_tf=Components(transforms=[
            BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args), is_random=True),
            AudioPadding(sample_rate=args.sample_rate, max_length=args.sample_rate, random_shift=False),
            ReduceChannel()
        ])
    )
    corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)
    accuracy = inference(args=args, auT=hubert, auC=clsModel, data_loader=corrupted_loader)
    print(f'Corrupted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(corrupted_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, args.background_type, args.corruption_level, accuracy, 100.-accuracy]

    records.to_csv(os.path.join(args.output_path, f'{args.background_type}-{args.corruption_level}.csv'))