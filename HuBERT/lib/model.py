import torch 
from torch import nn

class Classifier(nn.Module):
    def __init__(self, class_num:int, embed_size:int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=embed_size, out_features=embed_size//2)
        self.bn1 = nn.BatchNorm1d(num_features=embed_size//2)
        self.fc2 = nn.Linear(in_features=embed_size//2, out_features=class_num)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=1)
        x = self.bn1(self.fc1(x))
        x = self.fc2(x)
        return x