import argparse
import os

from torch import nn

def make_unless_exits(url:str) -> None:
    if not os.path.exists(url):
        os.makedirs(url)

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')

def count_ttl_params(model: nn.Module, filter_by_grad=False, requires_grad=True):
    if not filter_by_grad:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)

def store_model_structure_to_txt(model: nn.Module, output_path: str) -> None:
    model_info = str(model)
    with open(output_path, 'w') as f:
        f.write(model_info)