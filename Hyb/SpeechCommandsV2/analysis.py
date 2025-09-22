import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import Wav2Vec2Model

from lib.utils import make_unless_exits, print_argparse, count_ttl_params, ConfigDict
from lib.corruption import corruption_meta, SpeechCommandsV2C, CorruptionMeta
from lib.dataset import MultiTFDataset
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer, ReduceChannel
from AuT.SpeechCommandsV2.train import build_model as aut_build_model
from AuT.lib.model import FCEClassifier, AudioClassifier
from HuBERT.SpeechCommandsV2.train import build_model as hub_build_model
from Hyb.lib.utils import load_weight, merg_outs, config

def inference(
    args:argparse.Namespace, aut:FCEClassifier, aut_clsf:AudioClassifier,
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
        _, preds = torch.max(outs.detach(), dim=1)

        if idx == 0:
            ttl_corr = (preds == labels).sum().cpu().item()
            ttl_size = labels.shape[0]
        else:
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
    return ttl_corr / ttl_size

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--aut_wght_pth', type=str)
    ap.add_argument('--hub_wght_path', type=str)
    ap.add_argument('--adpt_aut_wght_pth', type=str)
    ap.add_argument('--adpt_hub_wght_pth', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'Hyb'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    args.config = ConfigDict()
    config(cfg=args.config, aut_rate=0.7761, hub_rate=0.9568, softmax=True, tag='WHN_L1')
    config(cfg=args.config, aut_rate=0.1, hub_rate=1.0, softmax=True, tag='WHN_L2')
    config(cfg=args.config, aut_rate=0.5694, hub_rate=0.9439, softmax=True, tag='ENQ_L1')
    config(cfg=args.config, aut_rate=0.4789, hub_rate=0.9495, softmax=False, tag='ENQ_L2')
    config(cfg=args.config, aut_rate=0.3, hub_rate=1.0, softmax=True, tag='END1_L1')
    config(cfg=args.config, aut_rate=0.8198, hub_rate=0.9650, softmax=True, tag='END1_L2')
    config(cfg=args.config, aut_rate=0.7235, hub_rate=0.9677, softmax=True, tag='END2_L1')
    config(cfg=args.config, aut_rate=0.6159, hub_rate=0.9635, softmax=True, tag='END2_L2')
    config(cfg=args.config, aut_rate=0.6616, hub_rate=0.9383, softmax=True, tag='ENSC_L1')
    config(cfg=args.config, aut_rate=0.7880, hub_rate=0.9368, softmax=True, tag='ENSC_L2')
    config(cfg=args.config, aut_rate=0.5147, hub_rate=0.9291, softmax=True, tag='PSH_L1')
    config(cfg=args.config, aut_rate=0.4593, hub_rate=0.9079, softmax=True, tag='PSH_L2')
    config(cfg=args.config, aut_rate=0.8412, hub_rate=0.9680, softmax=True, tag='TST_L1')
    config(cfg=args.config, aut_rate=0.7408, hub_rate=0.9679 , softmax=True, tag='TST_L2')

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
    args.target_length=104
    for idx, cmeta in enumerate(cmetas):
        print(f'{idx+1}/{len(cmetas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing ...')
        adpt_set = MultiTFDataset(
            dataset=SpeechCommandsV2C(
                root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type
            ),
            tfs=[
                Components(transforms=[
                    MelSpectrogram(
                        sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length,
                        hop_length=hop_length, mel_scale=mel_scale, n_mels=args.n_mels
                    ),
                    AmplitudeToDB(top_db=80., max_out=2.),
                    FrequenceTokenTransformer()
                ]),
                ReduceChannel()
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
        test_accu = inference(
            args=args, aut=aut, aut_clsf=aut_clsf, hub=hub, hub_clsf=hub_clsf, 
            data_loader=adpt_loader, cmeta=cmeta
        )

        print('Adaptation analysis ...')
        load_weight(args=args, embed=aut, clsf=aut_clsf, model='AMAuT', cmeta=cmeta, mode='adaptation')
        load_weight(args=args, embed=hub, clsf=hub_clsf, model='HuBERT', cmeta=cmeta, mode='adaptation')
        adpt_accu = inference(
            args=args, aut=aut, aut_clsf=aut_clsf, hub=hub, hub_clsf=hub_clsf,
            data_loader=adpt_loader, cmeta=cmeta
        )
        print(f'{args.dataset} {cmeta.type}-{cmeta.level} non-adapted accuracy is: {test_accu:.4f}, adapted accuracy is : {adpt_accu:.4f}, sample size is: {len(adpt_set)}')
        records.loc[len(records)] = [args.dataset, 'Hybrid', param_no, f'{cmeta.type}-{cmeta.level}', test_accu, adpt_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))