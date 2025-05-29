import random

import torch 
from torch import nn

class time_shift(nn.Module):
    def __init__(self, shift_limit: float, is_random=True, is_bidirection=False) -> None:
        """
        Time shift data augmentation

        :param shift_limit: shift_limit -> (-1, 1), shift_limit < 0 is left shift
        """
        super().__init__()
        self.shift_limit = shift_limit
        self.is_random = is_random
        self.is_bidirection = is_bidirection

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.is_random:
            shift_arg = int(random.random() * self.shift_limit * wavform.shape[1])
            if self.is_bidirection:
                shift_arg = int((random.random() * 2 - 1) * self.shift_limit * wavform.shape[1])
        else:
            shift_arg = int(self.shift_limit * wavform.shape[1])
        return wavform.roll(shifts=shift_arg)

class ReduceChannel(nn.Module):
    def __init__(self):
        super(ReduceChannel, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, dim=0)

class AudioPadding(nn.Module):
    def __init__(self, max_length:int, sample_rate:int, random_shift:bool=False):
        super(AudioPadding, self).__init__()
        self.max_length = max_length
        self.random_shift = random_shift

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        l = self.max_length - x.shape[1]
        if l > 0:
            if self.random_shift:
                head = random.randint(0, l)
                tail = l - head
            else:
                head = l // 2
                tail = l - head
            x = pad(x, (head, tail), mode='constant', value=0.)
        return x