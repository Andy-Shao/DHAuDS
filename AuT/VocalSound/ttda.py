import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.spdataset import VocalSound
from lib.dataset import MultiTFDataset, mlt_load_from, mlt_store_to, batch_store_to
from lib.component import Components, AudioPadding, AudioClip, DoNothing, time_shift
from lib.component import AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.lr_utils import build_optimizer, lr_scheduler
from lib.loss import entropy, g_entropy, nucnm
from AuT.VocalSound.train import build_model, inference
from AuT.ReefSet.ttda import load_weight

def vs_corruption_data(args:argparse.Namespace) -> Dataset:
    from lib.corruption import corrupt_data
    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    if args.corruption_type == 'TST':
        test_set = corrupt_data(
            orgin_set=VocalSound(
                root_path=args.dataset_root_path, mode='test', include_rate=False, version='16k',
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
        test_set = MultiTFDataset(
            dataset=test_set, 
            tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                ])
            ]
        )
    else:
        test_set = corrupt_data(
            orgin_set=VocalSound(
                root_path=args.dataset_root_path, mode='test', include_rate=False, version='16k',
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                ])
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    if args.corruption_type == 'PSH':
        mlt_store_to(
            dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name, 
            data_tfs=[DoNothing()]
        )
    else:
        batch_store_to(
            data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8),
            root_path=dataset_root_path, index_file_name=index_file_name, f_num=1
        )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ 
            Components(transforms=[
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        ]
    )
    adapt_set = MultiTFDataset(
        dataset=mlt_load_from(root_path=dataset_root_path, index_file_name=index_file_name),
        tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=True),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        ]
    )
    return test_set, adapt_set

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--aut_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--aut_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'TTDA')
    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    test_set, adapt_set = vs_corruption_data(args)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    aut, clsf = build_model(args)
    load_weight(args=args, aut=aut, clsf=clsf)
    optimizer = build_optimizer(lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)

    def inferecing(max_accu:float) -> tuple[float, float]:
        accuracy = inference(args=args, aut=aut, clsf=clsf, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(adapt_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            torch.save(
                aut.state_dict(), 
                os.path.join(args.output_path, f'aut-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}.pt')
            )
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'clsf-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}.pt')
            )
        return accuracy, max_accu
    
    max_accu = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
            
        print('Inferencing...')
        accuracy, max_accu = inferecing(max_accu)

        print('Adaptating...')
        aut.train(); clsf.train()
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.; ttl_gent_loss = 0.
        for fs1, fs2, _ in tqdm(adapt_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1 = clsf(aut(fs1)[1])
            os2 = clsf(aut(fs2)[1])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1, epsilon=1e-8) + entropy(args, os2, epsilon=1e-8)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os1, q=args.gent_q)

            loss = nucnm_loss + ent_loss + gent_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, 
                threshold=args.lr_threshold, momentum=args.lr_momentum
            )

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
    print('Finalizing...')
    accuracy, max_accu = inferecing(max_accu)
    wandb_run.log(
        data={
            'Loss/ttl_loss': ttl_loss / ttl_size,
            'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
            'Loss/Entropy loss': ttl_ent_loss / ttl_size,
            'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
            'Adaptation/accuracy': accuracy,
            'Adaptation/LR': learning_rate,
            'Adaptation/max_accu': max_accu,
        }, step=args.max_epoch, commit=True
    )