import argparse
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score

import torch 
from torch.utils.data import DataLoader
from torchaudio.models import Wav2Vec2Model
from torchaudio.transforms import Resample, MelSpectrogram

from lib.utils import make_unless_exits, print_argparse, count_ttl_params, ConfigDict
from lib.corruption import corruption_meta, UrbanSound8KC, CorruptionMeta
from lib.dataset import MultiTFDataset
from lib.component import Components, ReduceChannel, AudioClip, AmplitudeToDB, FrequenceTokenTransformer
from AuT.lib.model import FCETransform, AudioClassifier
from AuT.UrbanSound8K.train import build_model as aut_build_model
from HuBERT.lib.model import HuBClassifier
from HuBERT.UrbanSound8K.train import build_model as hub_build_model
from Hyb.lib.utils import load_weight, merg_outs, config

def inference(
    args:argparse.Namespace, aut:FCETransform, aut_clsf:AudioClassifier, hub:Wav2Vec2Model, 
    hub_clsf:HuBClassifier, data_loader:DataLoader, cmeta:CorruptionMeta
) -> float:
    aut.eval(); aut_clsf.eval()
    hub.eval(); hub_clsf.eval()
    for idx, (aut_fs, hub_fs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        aut_fs, hub_fs = aut_fs.to(args.device), hub_fs.to(args.device)
        labels = labels.to(args.device)

        with torch.inference_mode():
            aut_outs, _ = aut_clsf(aut(aut_fs)[0])
            hub_outs = hub_clsf(hub(hub_fs)[0])
        outs = merg_outs(args=args, aut_os=aut_outs, hub_os=hub_outs, cmeta=cmeta)
        _, preds = torch.max(outs.detach(), dim=1)

        if idx == 0:
            y_true = labels.detach().cpu()
            y_pred = preds.cpu()
        else:
            y_true = torch.cat([y_true, labels.detach().cpu()], dim=0)
            y_pred = torch.cat([y_pred, preds.cpu()], dim=0)
    return f1_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average='macro')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--aut_wght_pth', type=str)
    ap.add_argument('--hub_wght_path', type=str)
    ap.add_argument('--adpt_aut_wght_pth', type=str)
    ap.add_argument('--adpt_hub_wght_pth', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 44100
        args.aut_sample_rate = args.sample_rate
        args.hub_sample_rate = 16000
        args.aut_audio_length = int(4 * args.aut_sample_rate)
        args.hub_audio_length = int(4 * args.hub_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'Merg'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    args.config = ConfigDict()
    config(cfg=args.config, aut_rate=0.7231, hub_rate=0.6746, softmax=False, tag='WHN_L1')
    config(cfg=args.config, aut_rate=0.7093, hub_rate=0.6783, softmax=False, tag='WHN_L2')
    config(cfg=args.config, aut_rate=0.7137, hub_rate=0.6312, softmax=False, tag='ENSC_L1')
    config(cfg=args.config, aut_rate=0.7036, hub_rate=0.4766, softmax=False, tag='ENSC_L2')
    config(cfg=args.config, aut_rate=0.5392, hub_rate=0.6728, softmax=True, tag='PSH_L1')
    config(cfg=args.config, aut_rate=0.2866, hub_rate=0.6518, softmax=False, tag='PSH_L2')
    config(cfg=args.config, aut_rate=0.7078, hub_rate=0.6650, softmax=True, tag='TST_L1')
    config(cfg=args.config, aut_rate=0.7125, hub_rate=0.6525, softmax=True, tag='TST_L2')

    print_argparse(args)
    ##########################################
    corruption_types=['WHN', 'ENSC', 'PSH', 'TST'] 
    corruption_levels=['L1', 'L2']
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted'])
    cmetas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)

    args.n_mels=64
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=589
    for idx, cmeta in enumerate(corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)):
        print(f'{idx+1}/{len(cmetas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        adpt_set = MultiTFDataset(
            dataset=UrbanSound8KC(
                root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type,
            ),
            tfs=[
                Components(transforms=[
                    AudioClip(max_length=args.aut_audio_length, mode='head', is_random=False),
                    MelSpectrogram(
                        sample_rate=args.aut_sample_rate, n_fft=n_fft, win_length=win_length, 
                        hop_length=hop_length, n_mels=args.n_mels, mel_scale=mel_scale
                    ),
                    AmplitudeToDB(top_db=80., max_out=2.),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
                    Resample(orig_freq=args.sample_rate, new_freq=args.hub_sample_rate),
                    AudioClip(max_length=args.hub_audio_length, mode='head', is_random=False),
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
        records.loc[len(records)] = [args.dataset, param_no, 'Hybrid', f'{cmeta.type}-{cmeta.level}', test_accu, adpt_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))
