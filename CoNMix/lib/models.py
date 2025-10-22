import numpy as np

import torch 
import torch.nn as nn

from CoNMix.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from CoNMix.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1., .02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 100
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        self.feature_extractor = ViT_seg(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
        self.feature_extractor.load_from(weights=np.load(config_vit.pretrained_path))
        self.in_features = 2048
    
    def forward(self, x):
        _, feat = self.feature_extractor(x)
        return feat
    
class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=.5)
        self.bottleneck = nn.Linear(in_features=feature_dim, out_features=bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x: torch.Tensor):
        x = self.bottleneck(x)
        if self.type == 'bn':
            x = self.bn(x)
        return x
    
class feat_classifier(nn.Module):
    def __init__(self, class_num: int, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            # self.fc = nn.utils.weight_norm(nn.Linear(in_features=bottleneck_dim, out_features=class_num), name='weight')
            self.fc = nn.utils.parametrizations.weight_norm(module=nn.Linear(in_features=bottleneck_dim, out_features=class_num), name='weight')
            self.fc.apply(init_weights)
        else: 
            self.fc = nn.Linear(in_features=bottleneck_dim, out_features=class_num)
            self.fc.apply(init_weights)
    
    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x