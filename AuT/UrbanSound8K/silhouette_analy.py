import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib.utils import make_unless_exits, print_argparse
from lib.component import Components, AudioClip, AmplitudeToDB, FrequenceTokenTransformer
from lib.corruption import corruption_meta, UrbanSound8KC
from AuT.UrbanSound8K.train import build_model
from AuT.ReefSet.analysis import load_weight

def col_embeds(args:argparse.Namespace, aut:nn.Module, data_loader:DataLoader) -> torch.Tensor:
    aut.eval()
    for idx, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features = features.to(args.device)
        batch_size = labels.shape[0]

        with torch.inference_mode():
            ext_ft, _ = aut(features)
        
        ext_ft = ext_ft.cpu().detach()
        ext_ft = ext_ft.reshape(batch_size, -1)
        if idx == 0:
            ext_fts = ext_ft
            ttl_labels = labels.detach()
        else:
            ext_fts = torch.cat([ext_fts, ext_ft], dim=0)
            ttl_labels = torch.cat([ttl_labels, labels.detach()], dim=0)
    return ext_fts, ttl_labels

def analyzing(args:argparse.Namespace, corruption_types:list[str], corruption_levels:list[str]) -> None:
    args.n_mels=64
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=589
    records = pd.DataFrame(columns=['Alg.', 'Dataset', 'Corruption', 'Before', 'After', 'After - Before'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    aut, clsf = build_model(args=args)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')

        adpt_set = UrbanSound8KC(
            root_path=args.dataset_root_path, corruption_type=cmeta.type, corruption_level=cmeta.level,
            data_tf=Components(transforms=[
                AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    mel_scale=mel_scale, n_mels=args.n_mels
                ), # 64 x 589
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )
        
        print('Silhouette analysis before adaptation')
        load_weight(args=args, aut=aut, clsf=clsf, mode='origin')
        embeddings, labels = col_embeds(args=args, aut=aut, data_loader=adpt_loader)
        score_before = silhouette_score(X=embeddings.numpy(), labels=labels.numpy(), metric='euclidean')
        print('Silhouette analysis after adaptation')
        load_weight(args=args, aut=aut, clsf=clsf, mode='adaptation', metaInfo=cmeta)
        embeddings, labels = col_embeds(args=args, aut=aut, data_loader=adpt_loader)
        score_after = silhouette_score(X=embeddings.numpy(), labels=labels.numpy(), metric='euclidean')
        print(f'Silhouette Score (Before): {score_before:.4f}, Silhouette Score (After): {score_after:.4f}')
        records.loc[len(records)] = [args.arch, args.dataset, f'{cmeta.type}-{cmeta.level}', score_before, score_after, score_after - score_before]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 44100
        args.audio_length = int(4 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    analyzing(args=args, corruption_types=['WHN', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2']) 