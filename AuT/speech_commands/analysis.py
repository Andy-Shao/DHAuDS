import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
import torchaudio.transforms as a_transforms
from torch.utils.data import DataLoader

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.spdataset import SpeechCommandsV2, VocalSound, BackgroundNoiseDataset
from lib.dataset import RandomChoiceSet
from lib.component import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.component import BackgroundNoiseByFunc
from lib.acousticDataset import CochlScene
from AuT.speech_commands.train import build_model
from AuT.lib.model import FCETransform, AudioClassifier

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

def load_weight(args:argparse, mode:str, auT:FCETransform, auC:AudioClassifier) -> None:
    if mode == 'origin':
        auT_weight_path = args.origin_auT_weight
        cls_weight_path = args.origin_cls_weight
    auT.load_state_dict(state_dict=torch.load(auT_weight_path, weights_only=True))
    auC.load_state_dict(state_dict=torch.load(cls_weight_path, weights_only=True))

def noise_source(args:argparse):
    def expanding(noise: torch.Tensor, target_length=16000) -> torch.Tensor:
        noise_len = noise.shape[1]
        i = 1
        while i * noise_len < target_length: i += 1
        if i == 1: return noise
        else: return noise.repeat([1, i])

    if args.background_type == 'VocalSound':
        noise_set = VocalSound(root_path=args.background_path, mode='train', include_rate=False, version='16k')
    elif args.background_type == 'SCV2-BG':
        noise_set = BackgroundNoiseDataset(root_path=args.background_path, include_rate=False)
    elif args.background_type == 'CochlScene':
        noise_set = CochlScene(root_path=args.background_path, mode='train', include_rate=False)

    source = RandomChoiceSet(dataset=noise_set)
    return lambda: expanding(source[0][0])

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
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 'AuT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ###########################################################

    records = pd.DataFrame(columns=['dataset', 'module', 'param_num', 'corruption_type', 'corruption_level', 'accuracy', 'error_rate'])

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
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
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    print('Original')
    auTmodel, clsmodel = build_model(args)
    load_weight(args=args, mode='origin', auT=auTmodel, auC=clsmodel)
    param_num = count_ttl_params(model=auTmodel) + count_ttl_params(model=clsmodel)
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Original testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(test_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, args.background_type, args.corruption_level, accuracy, 100.-accuracy]

    print('Corrupted')
    corrupted_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='testing', download=True,
        data_tf=Components(transforms=[
            BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args), is_random=True),
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
    corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=corrupted_loader)
    print(f'Corrupted testing: accuracy is {accuracy:.4f}%, number of parameters is {param_num}, sample size is {len(corrupted_set)}')
    records.loc[len(records)] = [args.dataset, arch, param_num, args.background_type, args.corruption_level, accuracy, 100.-accuracy]

    for noise_type in ['doing_the_dishes', 'exercise_bike', 'running_tap']:
        pass

    records.to_csv(os.path.join(args.output_path, f'{args.background_type}-{args.corruption_level}.csv'))