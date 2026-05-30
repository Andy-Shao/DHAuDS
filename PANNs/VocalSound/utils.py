import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

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