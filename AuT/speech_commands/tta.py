import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.dataset import mlt_load_from, mlt_store_to
from lib.spdataset import SpeechCommandsV2, SpeechCommandsV1
from lib.component import Components, BackgroundNoiseByFunc, AudioPadding, AmplitudeToDB, MelSpectrogramPadding
from lib.component import FrequenceTokenTransformer, time_shift, DoNothing
from AuT.speech_commands.analysis import noise_source, load_weight, inference
from AuT.speech_commands.train import build_model, op_copy, lr_scheduler
from AuT.lib.model import FCETransform, AudioClassifier

def build_optimizer(args: argparse.Namespace, auT:FCETransform, auC:AudioClassifier) -> optim.Optimizer:
    param_group = []
    for _, v in auT.named_parameters():
        if args.auT_lr_decay > 0.:
            param_group +=  [{'params':v, 'lr': args.lr * args.auT_lr_decay}]
        else:
            v.requires_grad = False

    for _, v in auC.named_parameters():
        if args.auC_lr_decay > 0.:
            param_group +=  [{'params':v, 'lr': args.lr * args.auC_lr_decay}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--vocalsound_path', type=str)
    ap.add_argument('--cochlscene_path', type=str)
    ap.add_argument('--corruption_level', type=float)
    ap.add_argument('--corruption_type', type=str, choices=['doing_the_dishes', 'exercise_bike', 'running_tap', 'VocalSound', 'CochlScene'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--auT_lr_decay', type=float, default=1.)
    ap.add_argument('--auC_lr_decay', type=float, default=1.)
    ap.add_argument('--fbnm_rate', type=float, default=1.)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    elif args.dataset == 'SpeechCommandsV1':
        args.class_num = 30
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'TTA')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    arch = 'AuT'
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{arch}-{constants.dataset_dic[args.dataset]}-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    if args.dataset == 'SpeechCommandsV2':
        SpeechCommandsV2(root_path=args.dataset_root_path, mode='testing', download=True)
        corrupted_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=False, 
            data_tf=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True),
            ])
        )
    elif args.dataset == 'SpeechCommandsV1':
        corrupted_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='test',  include_rate=False,
            data_tfs=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True)
            ])
        )
    dataset_root_path = f'/root/tmp/{args.dataset}/{args.corruption_type}/{args.corruption_level}'
    index_file_name = 'metaInfo.csv'
    mlt_store_to(
        dataset=corrupted_set, 
        index_file_name=index_file_name, root_path=dataset_root_path,
        data_tfs=[DoNothing()]
    )

    corrupted_set = mlt_load_from(
        root_path=dataset_root_path, 
        index_file_name=index_file_name,
        data_tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
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

    corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name,
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
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    auTmodel, clsmodel = build_model(args)
    load_weight(args=args, mode='origin', auT=auTmodel, auC=clsmodel)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel)

    print('Pre-adaptation')
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Accurayc is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Adaptating...')
        auTmodel.train()
        clsmodel.train()
        ttl_size = 0.
        ttl_loss = 0.
        ttl_fbnm_loss = 0.
        for features, _ in tqdm(corrupted_loader):
            features = features.to(args.device)

            optimizer.zero_grad()
            outputs, _ = clsmodel(auTmodel(features)[0])

            # Nuclear-norm Maximization loss
            if args.fbnm_rate > 0:
                from torch.nn import functional as F
                softmax_outputs = F.softmax(input=outputs, dim=1)
                # list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_outputs,2),dim=0)), descending=True)
                # # require top class_num items, but if last batch size is lower than class_num, take batch size.
                # fbnm_loss = - torch.mean(list_svd[:min(softmax_outputs.shape[0], args.class_num)]) 
                fbnm_loss = - torch.mean(torch.sqrt(torch.sum(torch.pow(softmax_outputs,2),dim=0)))
                fbnm_loss = args.fbnm_rate * fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()

            loss = fbnm_loss
            loss.backward()
            optimizer.step()

            ttl_size += features.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_fbnm_loss += fbnm_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Inferencing...')
        accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')
        # if accuracy >= max_accu:
        #     max_accu = accuracy
        #     torch.save(auTmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-auT-{args.corruption_type}-{args.corruption_level}.pt'))
        #     torch.save(clsmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-cls-{args.corruption_type}-{args.corruption_level}.pt'))

        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_fbnm_loss / ttl_size,
                'Adaptation/accuracy': accuracy,
                'Adaptation/LR': learning_rate
            }, step=epoch, commit=True
        )
