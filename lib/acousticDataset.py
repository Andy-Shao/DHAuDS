import os
from dataclasses import dataclass
import pandas as pd

import torchaudio
from torch.utils.data import Dataset
from torch import nn
import torch

class ReefSet(Dataset):
    def __init__(
            self, root_path:str, mode:str, include_rate:bool=False, data_tf:nn.Module=None, label_tf:nn.Module=None,
            train_annotations_file:str='./data/ReefSet/train_annotations.csv', 
            test_annotations_file:str='./data/ReefSet/test_annotations.csv', label_mode:str='single'
        ):
        super().__init__()
        self.root_path = root_path
        assert mode in ['train', 'test'], 'No support'
        self.mode = mode
        self.include_rate = include_rate
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.meta_info = pd.read_csv(train_annotations_file) if mode == 'train' else pd.read_csv(test_annotations_file)
        self.label_dic = ReefSet.__label_dic__(label_mode)

    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, index):
        meta = self.meta_info.iloc[index]
        wavform, sample_rate = torchaudio.load(os.path.join(self.root_path, 'full_dataset', meta['file_name']), normalize=True)
        eye_matrix = torch.eye(len(self.label_dic), dtype=float)
        label = torch.zeros_like(eye_matrix[0], dtype=float)
        for k in self.label_dic[meta['label']]:
            label += eye_matrix[k]
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        if self.include_rate: 
            return wavform, label, sample_rate
        else:
            return wavform, label
    
    @staticmethod
    def __label_dic__(label_mode) -> dict:
        if label_mode == 'multiple':
            return {
            'ambient': [0], 

            'anthrop_boat_engine': [1],
            'anthrop_bomb': [2],
            'anthrop_mechanical': [3],

            'bioph': [4],
            'bioph_cascading_saw': [4, 5],
            'bioph_chatter': [4, 6],
            'bioph_chorus': [4, 7],
            'bioph_crackle': [4, 8],
            'bioph_croak': [4, 9],
            'bioph_damselfish': [4, 10],
            'bioph_dolphin': [4, 11],
            'bioph_double_pulse': [4, 12],
            'bioph_echinidae': [4, 13],
            'bioph_epigut': [4, 14],
            'bioph_grazing': [4, 15],
            'bioph_grouper_a': [4, 16],
            'bioph_grouper_groan': [4, 17],
            'bioph_growl': [4, 18],
            'bioph_holocentrus': [4, 19],
            'bioph_knock': [4, 20],
            'bioph_knock_croak_a': [4, 20, 21],
            'bioph_knock_croak_b': [4, 20, 22],
            'bioph_knock_croak_c': [4, 20, 23],
            'bioph_low_growl': [4, 24],
            'bioph_megnov': [4, 25],
            'bioph_midshipman': [4, 26],
            'bioph_mycbon': [4, 27],
            'bioph_pomamb': [4, 28],
            'bioph_pulse': [4, 29],
            'bioph_rattle': [4, 30],
            'bioph_rattle_response': [4, 31],
            'bioph_series_a': [4, 32],
            'bioph_series_b': [4, 33],
            'bioph_stridulation': [4, 34],
            'bioph_whup': [4, 35], 

            'geoph_waves': [36]
            }
        else:
            return {
            'ambient': [0], 

            'anthrop_boat_engine': [1],
            'anthrop_bomb': [2],
            'anthrop_mechanical': [3],

            'bioph': [4],
            'bioph_cascading_saw': [5],
            'bioph_chatter': [6],
            'bioph_chorus': [7],
            'bioph_crackle': [8],
            'bioph_croak': [9],
            'bioph_damselfish': [10],
            'bioph_dolphin': [11],
            'bioph_double_pulse': [12],
            'bioph_echinidae': [13],
            'bioph_epigut': [14],
            'bioph_grazing': [15],
            'bioph_grouper_a': [16],
            'bioph_grouper_groan': [17],
            'bioph_growl': [18],
            'bioph_holocentrus': [19],
            'bioph_knock': [20],
            'bioph_knock_croak_a': [21],
            'bioph_knock_croak_b': [22],
            'bioph_knock_croak_c': [23],
            'bioph_low_growl': [24],
            'bioph_megnov': [25],
            'bioph_midshipman': [26],
            'bioph_mycbon': [27],
            'bioph_pomamb': [28],
            'bioph_pulse': [29],
            'bioph_rattle': [30],
            'bioph_rattle_response': [31],
            'bioph_series_a': [32],
            'bioph_series_b': [33],
            'bioph_stridulation': [34],
            'bioph_whup': [35], 

            'geoph_waves': [36]
            }
        

class DEMAND(Dataset):
    def __init__(self, root_path:str, mode:str, include_rate:bool=False, data_tf:nn.Module=None):
        super().__init__()
        assert mode in [
            'DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY',
            'OMEETING', 'OOFFICE', 'PCAFETER', 'PRESTO', 'PSTATION', 'SPSQUARE', 'STRAFFIC',
            'TBUS', 'TCAR', 'TMETRO'
        ]
        self.mode = mode
        self.root_path = root_path
        self.include_rate = include_rate
        self.data_tf = data_tf 
        self.wav_list = self.__scan_wav__()
    
    def __scan_wav__(self) -> list[str]:
        wav_path = os.path.join(self.root_path, self.mode)
        ret = []
        for f in os.listdir(wav_path):
            if f.endswith('.wav'):
                ret.append(os.path.join(wav_path, f))
        return ret
    
    def __len__(self):
        return len(self.wav_list)
    
    def __getitem__(self, index):
        wav_path = self.wav_list[index]
        wavform, sample_rate = torchaudio.load(wav_path, normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.include_rate:
            return wavform, sample_rate
        else: return wavform

class QUTNOISE(Dataset):
    def __init__(self, root_path:str, mode:str, include_rate:bool=False, data_tf:nn.Module=None):
        super().__init__()
        assert mode in ['CAFE', 'CAR', 'HOME', 'REVERB', 'STREET']
        self.mode = mode
        self.root_path = root_path
        self.include_rate = include_rate
        self.data_tf = data_tf
        self.wav_list = self.__scan_wav__()

    def __scan_wav__(self) -> list[str]:
        wav_path = os.path.join(self.root_path, 'QUT-NOISE')
        ret = []
        for f in os.listdir(wav_path):
            if f.endswith('.wav') and f.startswith(self.mode):
                ret.append(os.path.join(wav_path, f))
        return ret

    def __len__(self):
        return len(self.wav_list)
    
    def __getitem__(self, index):
        wav_path = self.wav_list[index]
        wavform, sample_rate = torchaudio.load(wav_path, normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.include_rate:
            return wavform, sample_rate
        else:
            return wavform

class CochlScene(Dataset):
    """
    CochlScene is an acoustic scene classification dataset created using crowdsourcing. 
    It comprises approximately 76,000 audio samples across 13 distinct acoustic scenes. 
    The dataset was collected from 831 participants and features audio recordings of varied lengths, 
    with the majority ranging between 60 and 70 seconds.

    Sample rate: 44100, 10 seconds audio, mono audio

    The 13 acoustic scenes included in CochlScene are:
    Bus, Car, Elevator, Park, Restaurant, Street, SubwayStation, Cafe, CrowdedIndoor, Kitchen,
    ResidentialArea, Restroom, Subway
    """
    @dataclass
    class SampleMeta:
        sample_path: str
        label_name: str
        label_id: int
    label_dic = {
        'Bus': 0, 'Car': 1, 'Elevator': 2, 'Park': 3, 'Restaurant': 4, 'Street': 5, 'SubwayStation': 6,
        'Cafe': 7, 'CrowdedIndoor': 8, 'Kitchen': 9, 'ResidentialArea': 10, 'Restroom': 11, 'Subway': 12
    }
    mode_mapping = {
        'train': 'Train', 'validation': 'Val', 'test': 'Test'
    }
    def __init__(self, root_path:str, mode:str, data_tf:nn.Module=None, label_tf:nn.Module=None, include_rate:bool=False):
        super(CochlScene).__init__()
        assert mode in ['train', 'validation', 'test'], 'No support'
        self.root_path = root_path
        self.mode = mode
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.include_rate = include_rate

        self.data_list = self.__cal_data_list__()

    def __cal_data_list__(self) -> list[SampleMeta]:
        ret = []
        base_path = os.path.join(self.root_path, self.mode_mapping[self.mode])

        for k,v in self.label_dic.items():
            parent_path = os.path.join(base_path, k)
            for file_path in os.listdir(parent_path):
                if not file_path.endswith('.wav'):
                    continue
                meta_info = self.SampleMeta(
                    sample_path=os.path.join(parent_path, file_path),
                    label_name=k,
                    label_id=v
                )
                ret.append(meta_info)
        return ret

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        meta_info = self.data_list[index]
        wavform, sample_rate = torchaudio.load(meta_info.sample_path)
        label = meta_info.label_id

        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)

        if self.include_rate:
            return wavform, label, sample_rate
        return wavform, label