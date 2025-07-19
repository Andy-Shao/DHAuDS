import random

import torch
from torch import nn
from torchaudio.functional import add_noise, pitch_shift

class WHN(nn.Module):
    def __init__(self, lsnr:float, rsnr:float, step:float):
        super().__init__()
        self.lsnr = lsnr
        self.rsnr = rsnr
        self.step = step

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        channel, length = wavform.size()
        # for nt in ['Guassian', 'Uniform']:
        #     if nt == 'Guassian':
        #         noise = torch.normal(mean=0., std=1., size=[channel, length])
        #     elif nt == 'Uniform':
        #         noise = torch.rand([channel, length])
        #     snr_num = int((self.rsnr - self.lsnr)/self.step)
        #     wavform = add_noise(
        #         waveform=wavform, noise=noise,
        #         snr=torch.tensor([self.lsnr + (random.randint(0, snr_num) * self.step)])
        #     )
        nt = random.randint(0, 1)
        if nt == 0: # Guassian
            noise = torch.normal(mean=0., std=1., size=[channel, length])
        else: # Uniform
            noise = torch.rand([channel, length])
        snr_num = int((self.rsnr - self.lsnr)/self.step)
        wavform = add_noise(
            waveform=wavform, noise=noise,
            snr=torch.tensor([self.lsnr + (random.randint(0, snr_num) * self.step)])
        )
        return wavform

class DynEN(nn.Module):
    def __init__(self, noise_list:list[torch.Tensor], lsnr:float, rsnr:float, step:float):
        super().__init__()
        self.noise_list = noise_list
        self.lsnr = lsnr
        self.rsnr = rsnr
        self.step = step

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        channel, length = wavform.size()
        noise = self.noise_list[random.randint(0, len(self.noise_list) - 1)]
        start = random.randint(0, noise.shape[1]-length)
        noise = noise[:, start:start+length]
        if self.rsnr == self.lsnr:
            corr_wav = add_noise(waveform=wavform, noise=noise, snr=torch.tensor([self.lsnr]))
        else: 
            snr_num = int((self.rsnr - self.lsnr)/self.step)
            corr_wav = add_noise(
                waveform=wavform, noise=noise, snr=torch.tensor([self.lsnr + (random.randint(0, snr_num) * self.step)])
            )
        return corr_wav

class DynPSH(nn.Module):
    def __init__(self, sample_rate:int, min_steps:int, max_steps:int, is_bidirection:bool=False):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.is_bidirection = is_bidirection
        self.sample_rate = sample_rate

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        step = random.randint(self.min_steps, self.max_steps)
        if self.is_bidirection and random.randint(0, 1) == 1:
            step = - step
        return pitch_shift(waveform=wavform, sample_rate=self.sample_rate, n_steps=step)