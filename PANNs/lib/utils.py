import argparse
import os
from typing import Literal
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import ConfigDict
from lib.corruption import CorruptionMeta
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

def __cal_model_path__(
    args:argparse.Namespace, mode:Literal['origin', 'adaptation']='origin', 
    metaInfo:CorruptionMeta=None, root_path:str=None
) -> tuple[str, str]:
    if mode == 'origin':
        if root_path is None: root_path = args.orig_wght_pth
        h_p = os.path.join(root_path, f'panns-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        if root_path is None: root_path = args.adpt_wght_pth
        h_p = os.path.join(root_path, f'panns-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    return h_p, c_p

def load_weight(
    args:argparse.Namespace, panns:nn.Module, clsf:nn.Module, 
    mode:Literal['origin', 'adaptation']='origin', metaInfo:CorruptionMeta=None, root_path:str=None
) -> None:
    h_p, c_p = __cal_model_path__(args=args, mode=mode, metaInfo=metaInfo, root_path=root_path)
    panns.load_state_dict(state_dict=torch.load(h_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def store_weight(
    args:argparse.Namespace, panns:nn.Module, clsf:nn.Module, 
    mode:Literal['origin', 'adaptation']='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    torch.save(obj=panns.state_dict(), f=a_p)
    torch.save(obj=clsf.state_dict(), f=c_p)

def pan_freeze(pan:Wavegram_Logmel_Cnn14, batch1d:bool, batch2d:bool) -> None:
    for component in pan.modules():
        if isinstance(component, nn.BatchNorm1d) and batch1d:
            component.eval()
        elif isinstance(component, nn.BatchNorm2d) and batch2d:
            component.eval()

def build_model(args:argparse.Namespace, use_pre_weight:bool=True) -> tuple[Wavegram_Logmel_Cnn14, PANClassifier]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pan = Wavegram_Logmel_Cnn14(
        sample_rate=32000, 
        window_size=1024,
        hop_size=320,
        mel_bins=64, 
        fmin=50,
        fmax=14000,
        classes_num=527 #audio set
    )
    if use_pre_weight:
        ckpt = hf_hub_download(
            repo_id='nicofarr/panns_Wavegram_Logmel_Cnn14',
            filename='model.safetensors'
        )
        state = load_file(ckpt)
        # Remove 'backbone.' prefix
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        pan.load_state_dict(state_dict=state)
    cfg = ConfigDict()
    cfg.class_num = args.class_num
    cfg.embed_num = 2048
    clsf = PANClassifier(config=cfg)

    pan, clsf = pan.to(device=args.device), clsf.to(device=args.device)
    return pan, clsf

def inference(
    args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:PANClassifier, data_loader:DataLoader
) -> float:
    pan.eval(); clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    for features, labels in tqdm(data_loader):
        features = features.to(args.device)

        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
            outputs = outputs.detach().cpu()
        _, preds = torch.max(outputs, dim=1)
        ttl_corr += (preds == labels).sum().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size