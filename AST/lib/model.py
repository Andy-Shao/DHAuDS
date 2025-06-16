from torch import nn
import torch

from lib.utils import ConfigDict

class ASTClssifier(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(ASTClssifier, self).__init__()
        embed_size = config.embedding['embed_size']

        self.norm = nn.LayerNorm(normalized_shape=embed_size, eps=1e-6)
        self.fc = nn.Linear(in_features=embed_size, out_features=config.classifier['class_num'])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc(x)

        return x