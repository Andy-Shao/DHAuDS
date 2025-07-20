import argparse

import torch

def g_entropy(args:argparse.Namespace, outputs:torch.Tensor, q:float=.9) -> torch.Tensor:
    """
    " Generalized Entropy loss
    """
    if args.gent_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        gent_loss = (1 - torch.sum(torch.pow(softmax_outputs, exponent=q), dim=1)) / (q - 1)
        gent_loss = torch.mean(gent_loss)
        gent_loss = args.gent_rate * gent_loss
    else:
        gent_loss = torch.tensor(.0).to(args.device)
    return gent_loss

def entropy(args:argparse.Namespace, outputs:torch.Tensor, epsilon:float=1e-6) -> torch.Tensor:
    """
    " Entropy loss
    """
    if args.ent_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        ent_loss = - softmax_outputs * torch.log(softmax_outputs + epsilon)
        ent_loss = torch.mean(torch.sum(ent_loss, dim=1), dim=0)
        ent_loss = args.ent_rate * ent_loss
    else:
        ent_loss = torch.tensor(.0).to(args.device)
    return ent_loss

def nucnm(args:argparse.Namespace, outputs:torch.Tensor) -> torch.Tensor:
    """
    " Nuclear-norm Maximization loss with Frobenius Norm
    """
    if args.nucnm_rate > 0:
        from torch.nn import functional as F
        softmax_outputs = F.softmax(input=outputs, dim=1)
        nucnm_loss = - torch.mean(torch.sqrt(torch.sum(torch.pow(softmax_outputs,2),dim=0)))
        nucnm_loss = args.nucnm_rate * nucnm_loss
    else:
        nucnm_loss = torch.tensor(.0).to(args.device)
    return nucnm_loss