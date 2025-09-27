import argparse

import torch

def js_entropy(args:argparse.Namespace, out1:torch.Tensor, out2:torch.Tensor, epsilon:float=1e-8) -> torch.Tensor:
    """
    " Jensen-Shannon entropy
    """
    def kl(left:torch.Tensor, right:torch.Tensor) -> torch.Tensor:
        return - left * torch.log(left/(right + epsilon))

    if args.js_rate > 0:
        from torch.nn import functional as F
        sft_out1 = F.softmax(out1, dim=1)
        sft_out2 = F.softmax(out2, dim=1)
        js_loss = 0.5 * (kl(left=sft_out1, right=sft_out2) + kl(left=sft_out2, right=sft_out1))
        js_loss = torch.mean(torch.sum(js_loss, dim=1), dim=0)
        js_loss = args.js_rate * js_loss
    else:
        js_loss = torch.tensor(0.).to(args.device)
    return js_loss

def mse(args:argparse.Namespace, out1:torch.Tensor, out2:torch.Tensor) -> torch.Tensor:
    """
    " Mean Squared Error
    """
    if args.mse_rate > 0:
        from torch.nn import functional as F
        sft_out1 = F.softmax(out1, dim=1)
        sft_out2 = F.softmax(out2, dim=1)
        l2_norm = torch.sqrt(torch.sum(torch.pow(sft_out1 - sft_out2, 2.0), dim=1))
        mse_loss = torch.mean(l2_norm, dim=0)
        mse_loss = args.mse_rate * mse_loss
    else:
        mse_loss = torch.tensor(0.).to(args.device)
    return mse_loss

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
