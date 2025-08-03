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

class DynTST(nn.Module):
    def __init__(
        self, min_rate:float, max_rate:float, step:float, is_bidirection:bool=False, 
        n_fft:int=1024, hop_length:int=256
    ):
        super().__init__()
        self.is_bidirection = is_bidirection
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.step = step
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        from torchaudio.transforms import TimeStretch, Spectrogram, GriffinLim
        rate_num = int((self.max_rate - self.min_rate) / self.step)
        rate = self.min_rate + (random.randint(0, rate_num) * self.step)
        if self.is_bidirection and random.random() > .5:
            rate = 1.0 - rate
        else:
            rate += 1.0
        spectrogram = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        spec = spectrogram(wavform)
        time_stretch = TimeStretch(n_freq=self.n_fft//2 + 1, hop_length=self.hop_length)
        str_spec = time_stretch(spec, rate)
        g_lim = GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length)
        str_wav = g_lim(torch.abs(str_spec))

        return str_wav