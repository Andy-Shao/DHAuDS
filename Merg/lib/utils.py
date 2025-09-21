import argparse
import os

import torch
from torch import nn

from lib import constants
from lib.corruption import CorruptionMeta

def load_weight(args:argparse.Namespace, embed:nn.Module, clsf:nn.Module, mode:str, cmeta:CorruptionMeta) -> None:
    assert mode in ['AMAuT', 'HuBERT']
    if mode == 'AMAuT':
        base_path = args.aut_wght_pth
        embed_path = os.path.join(base_path, f'aut-{constants.dataset_dic[args.dataset]}-{cmeta.type}-{cmeta.level}.pt')
        clsf_path = os.path.join(base_path, f'clsf-{constants.dataset_dic[args.dataset]}-{cmeta.type}-{cmeta.level}.pt')
    elif mode == 'HuBERT':
        base_path = args.hub_wght_path
        embed_path = os.path.join(base_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-{cmeta.type}-{cmeta.level}.pt')
        clsf_path = os.path.join(base_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-{cmeta.type}-{cmeta.level}.pt')
    
    embed.load_state_dict(state_dict=torch.load(embed_path, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(clsf_path, weights_only=True))

def merg_outs(
    args:argparse, aut_os:torch.Tensor, hub_os:torch.Tensor, cmeta:CorruptionMeta,
    softmax:bool=False,
) -> torch.Tensor:
    import torch.nn.functional as F
    if softmax:
        aut_os = F.softmax(aut_os, dim=1)
        hub_os = F.softmax(hub_os, dim=1)
    aut_rate = args.aut_rates[f'{cmeta.type}-{cmeta.level}']
    hub_rate = args.hub_rates[f'{cmeta.type}-{cmeta.level}']
    aut_os = (aut_rate/(aut_rate + hub_rate)) * aut_os
    hub_os = (hub_rate/(aut_rate + hub_rate)) * hub_os
    return aut_os + hub_os