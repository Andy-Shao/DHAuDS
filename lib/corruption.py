import random
from dataclasses import dataclass
import pandas as pd
import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio
from torchaudio.functional import add_noise, pitch_shift
from torchaudio.transforms import Resample

from lib import constants
from lib.spdataset import SpeechCommandsBackgroundNoise, SpeechCommandsV2
from lib.acousticDataset import DEMAND, QUTNOISE
from lib.dataset import MergSet, MultiTFDataset, GpuMultiTFDataset
from lib.component import Components, Stereo2Mono, DoNothing

def corrupt_data(
        orgin_set:Dataset, corruption_level:str, corruption_type:str, enq_path:str, sample_rate:int,
        end_path:str, ensc_path:str, end_mode='16k'
    ) -> Dataset:
    assert end_mode in ['16k', '48k'], 'No support'
    if corruption_level == 'L1':
        snrs = constants.DYN_SNR_L1 if corruption_type != 'WHN' else constants.DYN_WHN_L1
        n_steps = constants.DYN_PSH_L1
        rates = constants.DYN_TST_L1
    elif corruption_level == 'L2':
        snrs = constants.DYN_SNR_L2 if corruption_type != 'WHN' else constants.DYN_WHN_L2
        n_steps = constants.DYN_PSH_L2
        rates = constants.DYN_TST_L2
    if corruption_type == 'WHN':
        test_set = MultiTFDataset(dataset=orgin_set, tfs=[WHN(lsnr=snrs[0], rsnr=snrs[2], step=snrs[1])])
    elif corruption_type == 'ENQ':
        noise_modes = constants.ENQ_NOISE_L2_LIST if corruption_level == 'L2' else constants.ENQ_NOISE_L1_LIST
        test_set = MultiTFDataset(dataset=orgin_set, tfs=[
            DynEN(noise_list=enq_noises(noise_modes=noise_modes, enq_path=enq_path, sample_rate=sample_rate), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
        ])
    elif corruption_type == 'END1':
        noise_modes = constants.END1_NOISE_L2_LIST if corruption_level == 'L2' else constants.END1_NOISE_L1_LIST
        if end_mode == '16k':
            n_ls = end_noises(noise_modes=noise_modes, end_path=end_path, sample_rate=sample_rate)
        elif end_mode == '48k':
            n_ls = end_noise_48k(noise_modes=noise_modes, end_path=end_path, sample_rate=sample_rate)
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=n_ls, lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif corruption_type == 'END2':
        noise_modes = constants.END2_NOISE_L2_LIST if corruption_level == 'L2' else constants.END2_NOISE_L1_LIST
        if end_mode == '16k':
            n_ls = end_noises(noise_modes=noise_modes, end_path=end_path, sample_rate=sample_rate)
        elif end_mode == '48k':
            n_ls = end_noise_48k(noise_modes=noise_modes, end_path=end_path, sample_rate=sample_rate)
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=n_ls, lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif corruption_type == 'ENSC':
        noise_modes = constants.ENSC_NOISE_L2_LIST if corruption_level == 'L2' else constants.ENSC_NOISE_L1_LIST
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynEN(noise_list=ensc_noises(noise_modes=noise_modes, ensc_path=ensc_path, sample_rate=sample_rate), lsnr=snrs[0], step=snrs[1], rsnr=snrs[2])
            ]
        )
    elif corruption_type == 'PSH':
        test_set = GpuMultiTFDataset(
            dataset=orgin_set, tfs=[
                DynPSH(sample_rate=sample_rate, min_steps=n_steps[0], max_steps=n_steps[1], is_bidirection=True)
            ]
        )
    elif corruption_type == 'TST':
        test_set = MultiTFDataset(
            dataset=orgin_set, tfs=[
                DynTST(min_rate=rates[0], step=rates[1], max_rate=rates[2], is_bidirection=True)
            ]
        )
    else:
        raise Exception('No support')
    return test_set

def ensc_noises(ensc_path:str, noise_modes:list[str], sample_rate:int=16000) -> list[torch.Tensor]:
    SpeechCommandsV2(root_path=ensc_path, mode='testing', download=True)
    if sample_rate == 16000:
        noise_set = SpeechCommandsBackgroundNoise(
            root_path=os.path.join(ensc_path, 'speech_commands_v0.02', 'speech_commands_v0.02'), 
            include_rate=False
        )
    else: 
        noise_set = SpeechCommandsBackgroundNoise(
            root_path=os.path.join(ensc_path, 'speech_commands_v0.02', 'speech_commands_v0.02'), include_rate=False,
            data_tf=Resample(orig_freq=16000, new_freq=sample_rate)
        )
    print('Loading noise files...')
    noises = []
    for noise, noise_type in tqdm(noise_set):
        if noise_type in noise_modes:
            noises.append(noise)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def end_noises(end_path:str, sample_rate:int, noise_modes:list[str] = ['DKITCHEN', 'NFIELD', 'OOFFICE', 'PRESTO', 'TCAR']) -> list[torch.Tensor]:
    noises = []
    print('Loading noise files...')
    demand_set = MergSet([
        DEMAND(
            root_path=end_path, mode=md, include_rate=False,
            data_tf=Resample(orig_freq=16000, new_freq=sample_rate) if sample_rate != 16000 else DoNothing()
        ) for md in noise_modes
    ])
    for wavform in tqdm(demand_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def end_noise_48k(end_path:str, sample_rate:int, noise_modes:list[str] = ['DKITCHEN', 'NFIELD', 'OOFFICE', 'PRESTO', 'TCAR']) -> list[torch.Tensor]:
    noises = []
    print('Loading noise files...')
    demand_set = MergSet([
        DEMAND(
            root_path=end_path, mode=md, include_rate=False,
            data_tf=Components(transforms=[
                Resample(orig_freq=48000, new_freq=sample_rate) if sample_rate != 48000 else DoNothing()
            ])
        ) for md in noise_modes])
    for wavform in tqdm(demand_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

def enq_noises(enq_path:str, sample_rate:int, noise_modes:list[str] = ['CAFE', 'HOME', 'STREET']) -> list[torch.Tensor]:
    background_path = enq_path
    noises = []
    print('Loading noise files...')
    qutnoise_set = MergSet([
        QUTNOISE(
            root_path=background_path, mode=md, include_rate=False,
            data_tf=Components(transforms=[
                Resample(orig_freq=48000, new_freq=sample_rate) if sample_rate != 48000 else DoNothing(),
                Stereo2Mono()
            ])
        ) for md in noise_modes
    ])
    for wavform in tqdm(qutnoise_set):
        noises.append(wavform)
    print(f'TTL noise size is: {len(noises)}')
    return noises

@dataclass
class CorruptionMeta:
    type:str
    level:str

def corruption_meta(corruption_types:list[str], corruption_levels:list[str]) -> list[CorruptionMeta]:
    ret = []
    for ctype in corruption_types:
        for l in corruption_levels:
            meta = CorruptionMeta(type=ctype, level=l)
            ret.append(meta)
    return ret

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
    
class ReefSetC(Dataset):
    meta_file = 'test_annotations.csv'
    def __init__(
        self, root_path:str, corruption_type:str, corruption_level:str, data_tf:nn.Module=None, label_tf:nn.Module=None, 
        label_mode:str='single'):
        from lib.acousticDataset import ReefSet
        super().__init__()
        assert label_mode in ['multiple', 'single']
        self.root_path = root_path
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.meta_infos = pd.read_csv(os.path.join(root_path, self.meta_file), header=0)
        self.label_dic = ReefSet.__label_dic__(label_mode)
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, index):
        meta = self.meta_infos.iloc[index]
        wavform, sample_rate = torchaudio.load(
            uri=os.path.join(self.root_path, self.corruption_type, self.corruption_level, meta['file_name']), normalize=True
        )
        eye_matrix = torch.eye(len(self.label_dic), dtype=float)
        label = torch.zeros_like(eye_matrix[0], dtype=float)
        for k in self.label_dic[meta['label']]:
            label += eye_matrix[k]
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label
    
class UrbanSound8KC(Dataset):
    meta_file = os.path.join('metadata', 'UrbanSound8K.csv')
    def __init__(self, root_path:str, corruption_type:str, corruption_level:str, data_tf:nn.Module=None, label_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.data_ls = pd.read_csv(os.path.join(root_path, self.meta_file), header=0)

    def __len__(self):
        return len(self.data_ls)
    
    def __getitem__(self, index):
        meta_info = self.data_ls.iloc[index]
        wavform, sample_rate = torchaudio.load(
            uri=os.path.join(
                self.root_path, 'audio', self.corruption_type, self.corruption_level, f'fold{meta_info['fold']}', 
                meta_info['slice_file_name']
            ), normalize=True
        )
        label = int(meta_info['classID'])
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label
    
class SpeechCommandsV2C(Dataset):
    meta_file = 'testing_list.txt'
    def __init__(self, root_path:str, corruption_level:str, corruption_type:str, data_tf:nn.Module=None, label_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.data_ls = self.__cal_data_list__()

    def __cal_data_list__(self):
        with open(os.path.join(self.root_path, SpeechCommandsV2C.meta_file), 'r') as f:
            data_list = f.readlines()
        return [it.strip() for it in data_list]

    def __len__(self):
        return len(self.data_ls)
    
    def __getitem__(self, index):
        meta_info = self.data_ls[index]
        label = int(SpeechCommandsV2.label_dict[meta_info.split('/')[0]])
        wavform, sample_rate = torchaudio.load(
            os.path.join(self.root_path, self.corruption_type, self.corruption_level, meta_info), 
            normalize=True
        )
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label

class VocalSoundC(Dataset):
    labe_dic_file = 'class_labels_indices_vs.csv'
    meta_file = os.path.join('datafiles', 'te.json')
    def __init__(self, root_path:str, corruption_level:str, corruption_type:str, data_tf:nn.Module=None, label_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.corruption_level = corruption_level
        self.corruption_type = corruption_type
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.label_dict = pd.read_csv(os.path.join(root_path, self.labe_dic_file), header=0)
        self.sample_list = self.__file_list__()
    
    def __file_list__(self):
        import json
        with open(os.path.join(self.root_path, self.meta_file)) as f:
            json_str = json.load(f)
        file_list = pd.json_normalize(json_str['data'])
        file_list['file_name'] = [str(it).split('/')[-1] for it in file_list['wav']]
        return file_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        meta_info = self.sample_list.iloc[index]
        label = int(pd.Series(self.label_dict[self.label_dict['mid'] == meta_info['labels']]['index']).item())
        wavform, sample_rate = torchaudio.load(
            os.path.join(self.root_path, 'audio_16k', self.corruption_type, self.corruption_level, meta_info['file_name']),
            normalize=True
        )
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label