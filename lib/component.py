import random
import numpy as np

import torch 
from torch import nn
# from transformers import AutoFeatureExtractor

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

class Components(nn.Module):
    def __init__(self, transforms: list) -> None:
        super().__init__()
        assert transforms is not None, 'No support'
        self.transforms = nn.ModuleList(transforms)

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            wavform = transform(wavform)
        return wavform

class AmplitudeToDB(nn.Module):
    def __init__(self, top_db:float, max_out:float) -> None:
        from torchaudio import transforms
        super(AmplitudeToDB, self).__init__()
        self.model = transforms.AmplitudeToDB(top_db=top_db)
        self.max_out = max_out
        self.top_db = top_db

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x) / (self.top_db // self.max_out)

class MelSpectrogramPadding(nn.Module):
    def __init__(self, target_length):
        super(MelSpectrogramPadding, self).__init__()
        self.target_length = target_length

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        p = self.target_length - x.shape[2]
        if p > 0:
            # padding = nn.ZeroPad1d((0, p, 0, 0))
            # x = padding(x)
            x = pad(x, (0, p, 0, 0), mode='constant', value=0.)
        elif p < 0:
            x = x[:, :, 0:self.target_length]
        return x

class FrequenceTokenTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        c, token_num, token_len = x.size()
        x = x.reshape(-1, token_len)
        return x

class GuassianNoise(nn.Module):
    def __init__(self, noise_level=.05):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        ## Guassian Noise
        noise = torch.rand_like(wavform) * self.noise_level
        noise_wavform = wavform + noise
        return noise_wavform

class BackgroundNoise(nn.Module):
    def __init__(self, noise_level: float, noise: torch.Tensor, is_random=False):
        super().__init__()
        self.noise_level = noise_level
        self.noise = noise
        self.is_random = is_random

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as ta_f
        wav_len = wavform.shape[1]
        if self.is_random:
            start_point = np.random.randint(low=0, high=self.noise.shape[1]-wav_len)
            noise_period = self.noise[:, start_point:start_point+wav_len]
        else:
            noise_period = self.noise[:, 0:wav_len]
        noised_wavform = ta_f.add_noise(waveform=wavform, noise=noise_period, snr=torch.tensor([self.noise_level]))
        return noised_wavform
    
class BackgroundNoiseByFunc(nn.Module):
    def __init__(self, noise_level:float, noise_func, is_random:bool=False):
        super().__init__()
        self.noise_level = noise_level
        self.noise_func = noise_func
        self.is_random = is_random

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as ta_f
        wav_len = waveform.shape[1]
        noise = self.noise_func()
        noise_len = noise.shape[1]
        if wav_len == noise_len:
            noise_period = noise
        elif wav_len > noise_len:
            raise Exception(f'wav_len({wav_len}) is greater than noise_len({noise_len})')
        elif self.is_random:
            start_point = np.random.randint(low=0, high=noise.shape[1]-wav_len)
            noise_period = noise[:, start_point:start_point+wav_len]
        else:
            noise_period = noise[:, 0:wav_len]
        corrupted_waveform = ta_f.add_noise(waveform=waveform, noise=noise_period, snr=torch.tensor([self.noise_level]))
        return corrupted_waveform

class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x
    
# class ASTFeatureExt(nn.Module):
#     def __init__(self, feature_extractor:AutoFeatureExtractor, sample_rate:int):
#         super().__init__()
#         self.feature_extractor = feature_extractor
#         self.sample_rate = sample_rate

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         x = self.feature_extractor(x.numpy(), sampling_rate=self.sample_rate, return_tensors="pt", padding=False)['input_values']
#         return x