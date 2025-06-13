import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader

from lib.utils import make_unless_exits, print_argparse
from lib import constants
from lib.spdataset import SpeechCommandsV1, SpeechCommandsV2
from lib.component import Components, AudioPadding, ReduceChannel, BackgroundNoiseByFunc, DoNothing
from lib.dataset import mlt_load_from, mlt_store_to
from AuT.speech_commands.analysis import noise_source, load_weight
from AuT.speech_commands.tta import build_optimizer, nucnm, entropy, g_entropy
from AuT.speech_commands.train import lr_scheduler
from HuBERT.speech_commands.train import build_model
from HuBERT.speech_commands.analysis import inference

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
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--auT_lr_decay', type=float, default=1.)
    ap.add_argument('--auC_lr_decay', type=float, default=1.)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
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
    arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'TTA')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[arch]}-{constants.dataset_dic[args.dataset]}-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    sample_rate = 16000
    max_length = sample_rate
    if args.dataset == 'SpeechCommandsV1':
        corrupted_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='test', include_rate=False,
            data_tfs=Components(transforms=[
                AudioPadding(max_length=max_length, sample_rate=sample_rate, random_shift=False),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True),
            ])
        )
    elif args.dataset == 'SpeechCommandsV2':
        SpeechCommandsV2(root_path=args.dataset_root_path, mode='testing', download=True)
        corrupted_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=False,
            data_tf=Components(transforms=[
                AudioPadding(max_length=max_length, sample_rate=sample_rate, random_shift=False),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True),
            ])
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset, args.corruption_type, str(args.corruption_level))
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
                ReduceChannel()
            ])
        ]
    )
    corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    test_set = mlt_load_from(
        root_path=dataset_root_path,
        index_file_name=index_file_name,
        data_tfs=[
            Components(transforms=[
                ReduceChannel()
            ])
        ]
    )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    hubert, clsf = build_model(args=args, pre_weight=True)
    load_weight(args=args, mode='origin', auT=hubert, auC=clsf)
    optimizer = build_optimizer(args=args, auT=hubert, auC=clsf)

    print('Pre-adaptation')
    accuracy = inference(args=args, hubert=hubert, clsmodel=clsf, data_loader=test_loader)
    print(f'Accurayc is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Adaptating...')
        hubert.train(); clsf.train()
        ttl_size = 0.; ttl_loss = 0.
        ttl_nucnm_loss = 0.; ttl_ent_loss = 0.; ttl_gent_loss = 0.
        for fs, _ in tqdm(corrupted_loader):
            fs = fs.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsf(hubert(fs)[0])

            nucnm_loss = nucnm(args, os1)
            ent_loss = entropy(args, os1)
            gent_loss = g_entropy(args, os1, q=args.gent_q)

            loss = nucnm_loss + ent_loss + gent_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()
        
        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Inferencing...')
        accuracy = inference(args=args, hubert=hubert, clsmodel=clsf, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')

        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(hubert.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-hubert-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}{args.file_suffix}.pt'))
            torch.save(clsf.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-clsModel-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}{args.file_suffix}.pt'))
        
        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/Entropy loss': ttl_ent_loss / ttl_size,
                'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
                'Adaptation/accuracy': accuracy,
                'Adaptation/LR': learning_rate,
                'Adaptation/max_accu': max_accu,
            }, step=epoch, commit=True
        )