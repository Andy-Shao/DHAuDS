import torch 
from torch import nn

class VsAuT(nn.Module):
    def __init__(self, aut:nn.Module, clsf:nn.Module):
        super().__init__()
        self.aut = aut
        self.clsf = clsf

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        outputs = self.clsf(self.aut(x)[1])
        return outputs