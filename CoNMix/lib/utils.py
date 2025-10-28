import random
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class pad_trunc(nn.Module):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    """
    def __init__(self, max_ms: float, sample_rate: int) -> None:
        super().__init__()
        assert max_ms > 0, 'max_ms must be greater than zero'
        assert sample_rate > 0, 'sample_rate must be greater than zeror'
        self.max_ms = max_ms
        self.sample_rate = sample_rate

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        channel_num, wav_len = wavform.shape
        max_len = self.sample_rate // 1000 * self.max_ms

        if (wav_len > max_len):
            wavform = wavform[:, :max_len]
        elif wav_len < max_len:
            head_len = random.randint(0, max_len - wav_len)
            tail_len = max_len - wav_len - head_len

            head_pad = torch.zeros((channel_num, head_len))
            tail_pad = torch.zeros((channel_num, tail_len))

            wavform = torch.cat((head_pad, wavform, tail_pad), dim=1)
        return wavform

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
    
class Components(nn.Module):
    def __init__(self, transforms: list) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.transforms is None:
            return None
        for transform in self.transforms:
            wavform = transform(wavform)
        return wavform

class ExpandChannel(nn.Module):
    def __init__(self, out_channel: int) -> None:
        super().__init__()
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat((self.out_channel, 1, 1))

def cal_norm(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    for idx, (features, _) in tqdm(enumerate(loader), total=len(loader)):
        channel_size = features.shape[1]
        if idx == 0:
            mean = torch.zeros((channel_size), dtype=torch.float32)
            std = torch.zeros((channel_size), dtype=torch.float32)
        features = torch.transpose(features, 1, 0)
        features = features.reshape(channel_size, -1)
        mean += features.mean(1)
        std += features.std(1)
    mean /= len(loader)
    std /= len(loader)
    return mean.detach().numpy(), std.detach().numpy()

class Dataset_Idx(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int, int]:
        feature, label = self.dataset[index]
        return feature, label, index