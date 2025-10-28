import torch
import torch.nn.functional as F

def soft_CE(softout: torch.Tensor, soft_label: torch.Tensor, epsilon = 1e-5) -> torch.Tensor:
    """(Consist loss -> soft cross-entropy loss) uses this loss function"""
    # epsilon = 1e-5
    loss = -soft_label * torch.log(softout + epsilon)
    total_loss = torch.sum(loss, dim=1)
    return total_loss

def Entropy(input_: torch.Tensor, epsilon= 1e-5):
    # epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def SoftCrossEntropyLoss(logit: torch.Tensor, soft_pseudo_label: torch.Tensor) -> torch.Tensor:   # Checked and is correct
    """Pseudo-label cross-entropy loss uses this loss function"""
    percentage = F.log_softmax(logit, dim=1)
    # print(f'left shape: {soft_pseudo_label.shape}, right shape: {percentage.shape}')
    return -(soft_pseudo_label * percentage).sum(dim=1)