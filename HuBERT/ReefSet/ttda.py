import argparse
import wandb
import numpy as np
import random
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.acousticDataset import ReefSet
from lib.dataset import MultiTFDataset, batch_store_to, mlt_load_from, mlt_store_to
from lib.component import AudioPadding, ReduceChannel, Components, AudioClip, DoNothing, time_shift
from lib.lr_utils import build_optimizer, lr_scheduler
from lib.loss import nucnm, g_entropy, entropy, js_entropy, mse
from HuBERT.VocalSound.ttda import load_weigth
from HuBERT.ReefSet.train import build_model, inference

def rs_corrupt_data(args:argparse.Namespace) -> tuple[Dataset, Dataset]:
    from lib.corruption import corrupt_data
    if args.corruption_type == 'TST':
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
        test_set = MultiTFDataset(
            dataset=test_set, tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ])
            ]
        )
    else: 
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ])
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset)
    index_file_name = 'metaInfo.csv'
    if args.corruption_type == 'PSH':
        mlt_store_to(
            dataset=test_set, root_path=dataset_root_path, index_file_name=index_file_name, data_tfs=[DoNothing()],
            is_one_hot_label=True
        )
    else:
        batch_store_to(
            data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8), 
            root_path=dataset_root_path, index_file_name=index_file_name, f_num=1, is_one_hot_label=True
        )
    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name, 
        data_tfs=[ ReduceChannel() ], is_one_hot_label=True, class_num=args.class_num
    )
    adapt_set = MultiTFDataset(
        dataset=mlt_load_from(
            root_path=dataset_root_path, index_file_name=index_file_name, is_one_hot_label=True, class_num=args.class_num
        ), tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                ReduceChannel()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                ReduceChannel()
            ])
        ]
    )
    return test_set, adapt_set
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
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
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--mse_rate', type=float, default=0.0)
    ap.add_argument('--js_rate', type=float, default=0.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_wght_pth', type=str)
    ap.add_argument('--clsf_wght_pth', type=str)

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
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
        name=f'{constants.architecture_dic[args.arch]}-{constants.hubert_level_dic[args.model_level]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    test_set, adapt_set = rs_corrupt_data(args)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )
    adapt_loader = DataLoader(
        dataset=adapt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )

    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weigth(args=args, hubert=hubert, clsf=clsf, mode='origin')
    optimizer = build_optimizer(lr=args.lr, auT=hubert, auC=clsf, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay)

    def inferecing(max_roc_auc:float) -> tuple[float, float]:
        val_roc_auc = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=test_loader)
        print(f'ROC-AUC is: {val_roc_auc:.4f}, sample size is: {len(adapt_set)}')
        if val_roc_auc >= max_roc_auc:
            max_roc_auc = val_roc_auc
            torch.save(
                hubert.state_dict(), 
                os.path.join(args.output_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt')
            )
            torch.save(
                clsf.state_dict(), 
                os.path.join(args.output_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}{args.file_suffix}.pt')
            )
        return val_roc_auc, max_roc_auc
    
    max_roc_auc = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
            
        print('Inferencing...')
        val_roc_auc, max_roc_auc = inferecing(max_roc_auc)

        print('Adapting...')
        hubert.train(); clsf.train()
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.; ttl_gent_loss = 0.; ttl_const_loss = 0.
        for fs1, fs2, _ in tqdm(adapt_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsf(hubert(fs1)[0])
            os2, _ = clsf(hubert(fs2)[0])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1, epsilon=1e-8) + entropy(args, os2, epsilon=1e-8)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os1, q=args.gent_q)
            const_loss = mse(args=args, out1=os1, out2=os2) + js_entropy(args=args, out1=os1, out2=os2, epsilon=1e-8)

            loss = nucnm_loss + ent_loss + gent_loss + const_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()
            ttl_const_loss += const_loss.cpu().item()

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
                'Loss/Consistency loss': ttl_const_loss / ttl_size,
                'Adaptation/ROC-AUC': val_roc_auc,
                'Adaptation/LR': learning_rate,
                'Adaptation/Max_ROC-AUC': max_roc_auc,
            }, step=epoch, commit=True
        )
    print('Finalizing...')
    val_roc_auc, max_roc_auc = inferecing(max_roc_auc)
    wandb_run.log(
        data={
            'Loss/ttl_loss': ttl_loss / ttl_size,
            'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
            'Loss/Entropy loss': ttl_ent_loss / ttl_size,
            'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
            'Loss/Consistency loss': ttl_const_loss / ttl_size,
            'Adaptation/ROC-AUC': val_roc_auc,
            'Adaptation/LR': learning_rate,
            'Adaptation/Max_ROC-AUC': max_roc_auc,
        }, step=args.max_epoch, commit=True
    )
