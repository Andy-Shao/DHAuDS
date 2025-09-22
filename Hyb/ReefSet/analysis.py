import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader
from torchaudio.models import Wav2Vec2Model
from torchaudio.transforms import MelSpectrogram

from lib.utils import make_unless_exits, print_argparse, ConfigDict, count_ttl_params
from lib.corruption import corruption_meta, ReefSetC, CorruptionMeta
from lib.dataset import MultiTFDataset
from lib.component import Components, AudioPadding, AudioClip, AmplitudeToDB, FrequenceTokenTransformer
from lib.component import ReduceChannel
from Hyb.lib.utils import config, load_weight, merg_outs
from AuT.ReefSet.train import build_model as aut_build_model
from AuT.lib.model import FCETransform, AudioClassifier
from HuBERT.ReefSet.train import build_model as hub_build_model

def inference(
    args:argparse.Namespace, aut:FCETransform, aut_clsf:AudioClassifier,
    hub:Wav2Vec2Model, hub_clsf:AudioClassifier, data_loader:DataLoader, 
    cmeta:CorruptionMeta
) -> float:
    aut.eval(); aut_clsf.eval()
    hub.eval(); hub_clsf.eval()
    for idx, (aut_fs, hub_fs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        aut_fs, hub_fs = aut_fs.to(args.device), hub_fs.to(args.device)
        labels = labels.to(args.device)

        with torch.inference_mode():
            aut_outs, _ = aut_clsf(aut(aut_fs)[0])
            hub_outs, _ = hub_clsf(hub(hub_fs)[0])
        outs = merg_outs(args=args, aut_os=aut_outs, hub_os=hub_outs, cmeta=cmeta)

        if idx == 0:
            y_true = labels.detach().cpu()
            y_score = outs.detach().cpu()
        else:
            y_true = torch.concat([y_true, labels.detach().cpu()], dim=0)
            y_score = torch.concat([y_score, outs.detach().cpu()], dim=0)
    return roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--aut_wght_pth', type=str)
    ap.add_argument('--hub_wght_path', type=str)
    ap.add_argument('--orig_aut_wght_pth', type=str)
    ap.add_argument('--orig_hub_wght_pth', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'Hyb'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    args.config = ConfigDict()
    config(cfg=args.config, aut_rate=0.8779, hub_rate=0.7716, softmax=True, tag='WHN_L1')
    config(cfg=args.config, aut_rate=0.8683, hub_rate=0.7746, softmax=True, tag='WHN_L2')
    config(cfg=args.config, aut_rate=0.7922, hub_rate=0.6870, softmax=True, tag='ENQ_L1')
    config(cfg=args.config, aut_rate=0.7951, hub_rate=0.7215, softmax=True, tag='ENQ_L2')
    config(cfg=args.config, aut_rate=0.9064, hub_rate=0.8134, softmax=True, tag='END1_L1')
    config(cfg=args.config, aut_rate=0.9008, hub_rate=0.8156, softmax=True, tag='END1_L2')
    config(cfg=args.config, aut_rate=0.8721, hub_rate=0.7874, softmax=True, tag='END2_L1')
    config(cfg=args.config, aut_rate=0.8620, hub_rate=0.7703, softmax=True, tag='END2_L2')
    config(cfg=args.config, aut_rate=0.8355, hub_rate=0.7180, softmax=True, tag='ENSC_L1')
    config(cfg=args.config, aut_rate=0.8077, hub_rate=0.7192, softmax=True, tag='ENSC_L2')
    config(cfg=args.config, aut_rate=0.8623, hub_rate=0.9033, softmax=True, tag='PSH_L1')
    config(cfg=args.config, aut_rate=0.8316, hub_rate=0.8779, softmax=True, tag='PSH_L2')
    config(cfg=args.config, aut_rate=0.9871, hub_rate=0.8853, softmax=True, tag='TST_L1')
    config(cfg=args.config, aut_rate=0.9835, hub_rate=0.8653 , softmax=True, tag='TST_L2')

    print_argparse(args)
    ##########################################
    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    corruption_levels=['L1', 'L2']
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted'])
    cmetas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)

    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195
    for idx, cmeta in enumerate(cmetas):
        print(f'{idx+1}/{len(cmetas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing ...')
        adpt_set = MultiTFDataset(
            dataset=ReefSetC(
                root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type
            ),
            tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                    MelSpectrogram(
                        sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                        mel_scale=mel_scale, n_mels=args.n_mels
                    ),
                    AmplitudeToDB(top_db=80., max_out=2.),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                    ReduceChannel()
                ])
            ]
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers
        )
        aut, aut_clsf = aut_build_model(args=args)
        hub, hub_clsf = hub_build_model(args=args)
        param_no = count_ttl_params(aut) + count_ttl_params(aut_clsf) + count_ttl_params(hub) + count_ttl_params(hub_clsf)

        print('Non-adaptation analysis ...')
        load_weight(args=args, embed=aut, clsf=aut_clsf, model='AMAuT', cmeta=cmeta, mode='origin')
        load_weight(args=args, embed=hub, clsf=hub_clsf, model='HuBERT', cmeta=cmeta, mode='origin')
        test_roc_auc = inference(
            args=args, aut=aut, aut_clsf=aut_clsf, hub=hub, hub_clsf=hub_clsf, 
            data_loader=adpt_loader, cmeta=cmeta
        )

        print('Adaptation analysis ...')
        load_weight(args=args, embed=aut, clsf=aut_clsf, model='AMAuT', cmeta=cmeta, mode='adaptation')
        load_weight(args=args, embed=hub, clsf=hub_clsf, model='HuBERT', cmeta=cmeta, mode='adaptation')
        adpt_roc_auc = inference(
            args=args, aut=aut, aut_clsf=aut_clsf, hub=hub, hub_clsf=hub_clsf,
            data_loader=adpt_loader, cmeta=cmeta
        )
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy is: {test_roc_auc:.4f}, adapted accuracy is : {adpt_roc_auc:.4f}, sample size is: {len(adpt_set)}')
        records.loc[len(records)] = [args.dataset, 'Hybrid', param_no, f'{cmeta.type}-{cmeta.level}', test_roc_auc, adpt_roc_auc]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))