import argparse
from tqdm import tqdm
import copy
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader

from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

def inference(args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:PANClassifier, data_loader:DataLoader) -> float:
    pan.eval(); clsf.eval()
    for i, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features = features.to(args.device)

        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
            outputs = outputs.detach().cpu()
        _, preds = torch.max(outputs, dim=1)
        if i == 0: 
            y_preds = [preds]
            y_trues = [copy.deepcopy(labels.detach())]
        else:
            y_preds.append(preds)
            y_trues.append(copy.deepcopy(labels.detach()))
    return f1_score(
        y_true=torch.concat(y_trues, dim=0).numpy(), y_pred=torch.concat(y_preds, dim=0).numpy(), 
        average='macro'
    )