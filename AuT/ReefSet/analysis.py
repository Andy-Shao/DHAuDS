import argparse
import os

import torch
from torch import nn

from lib import constants
from lib.corruption import CorruptionMeta

def load_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None
) -> None:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        a_p = os.path.join(args.orig_wght_pth, f'aut-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(args.orig_wght_pth, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        a_p = os.path.join(args.adpt_wght_path, f'aut-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(args.adpt_wght_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))