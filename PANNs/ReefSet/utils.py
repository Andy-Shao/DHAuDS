import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

def inference(
    args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:PANClassifier, data_loader:DataLoader
) -> float:
    pan.eval(); clsf.eval()
    for i, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features = features.to(args.device)

        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
            outputs = outputs.detach().cpu()
        if i == 0:
            y_t = [labels.detach()]
            y_s = [nn.functional.softmax(outputs, dim=1)]
        else:
            y_t.append(labels.detach())
            y_s.append(nn.functional.softmax(outputs, dim=1))
    eval_roc_auc = roc_auc_score(
        y_true=torch.concat(y_t, dim=0).numpy(), y_score=torch.concat(y_s, dim=0).numpy(),
        average='macro', multi_class='ovr'
    )
    return eval_roc_auc