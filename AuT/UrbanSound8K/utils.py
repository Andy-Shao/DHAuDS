import torch 
from torch import nn

class Us8AuT(nn.Module):
    def __init__(self, aut:nn.Module, clsf:nn.Module):
        super().__init__()
        self.aut = aut
        self.clsf = clsf

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        outputs, _ = self.clsf(self.aut(x)[0])
        return outputs