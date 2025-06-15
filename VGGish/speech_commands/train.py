import argparse
import os
import numpy as np
import random
import wandb

import torch
from torchaudio.prototype import pipelines

from VGGish.lib.model import AudioClassifier
from lib.utils import ConfigDict, print_argparse, make_unless_exits
from lib import constants

def build_model(args:argparse.Namespace) -> tuple[pipelines.VGGishBundle, pipelines.VGGishBundle.VGGish, AudioClassifier]:
    bundle = pipelines.VGGISH
    vgg = bundle.get_model().to(args.device)
    cfg = ConfigDict()
    cfg.embedding = ConfigDict()
    cfg.embedding['embed_size'] = 128
    cfg.classifier['extend_size'] = 256
    cfg.classifier['convergent_size'] = 128
    cfg.classifier['class_num'] = args.class_num
    clsmodel = AudioClassifier(config=cfg).to(args.device)
    return bundle, vgg, clsmodel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default='2025')
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    elif args.dataset == 'SpeechCommandsV1':
        args.class_num = 30
    else:
        raise Exception('No support!')
    args.sample_rate = 16000
    arch = 'VGGish'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'train')

    torch.backends.cudnn.benchmark == True
    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    print_argparse(args=args)
    ############################################################

    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )

    vgg_bundle, vgg, clsmodel = build_model(args)