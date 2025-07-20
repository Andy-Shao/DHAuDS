import os
from typing import Any
from dataclasses import dataclass
import pandas as pd

from torch.utils.data import Dataset
import torch 
import torchaudio

class SpeechCommandsV2(Dataset):
    label_dict = {
        'backward': 0, 'bed': 3, 'bird': 32, 'cat': 17, 'dog': 10, 'down': 14, 'eight': 13, 'five': 2, 
        'follow': 1, 'forward': 16, 'four': 20, 'go': 33, 'happy': 31, 'house': 8, 'learn': 6, 'left': 26, 
        'marvin': 27, 'nine': 23, 'no': 22, 'off': 24, 'on': 5, 'one': 34, 'right': 18, 'seven': 12, 
        'sheila': 30, 'six': 15, 'stop': 11, 'three': 25, 'tree': 9, 'two': 7, 'up': 29, 'visual': 19, 
        'wow': 21, 'yes': 28, 'zero': 4
    }
    def __init__(
            self, root_path:str, folder_in_archive:str='speech_commands_v0.02', mode:str=None, download:bool = True, 
            data_tf:torch.nn.Module=None, label_tf:torch.nn.Module=None
        ):
        from torchaudio.datasets import SPEECHCOMMANDS
        super(SpeechCommandsV2, self).__init__()
        assert mode in ['training', 'validation', 'testing', None], 'No support'

        if not os.path.exists(root_path):
            os.makedirs(root_path)  

        self.dataset = SPEECHCOMMANDS(
            root=root_path, url='speech_commands_v0.02', folder_in_archive=folder_in_archive, subset=mode, download=download
        )
        self.data_tf = data_tf
        self.label_tf = label_tf

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        wavform, sample_rate, label, speaker_id, utterance_num = self.dataset[index]
        label = int(self.label_dict[label])
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label

class SpeechCommandsBackgroundNoise(Dataset):
    base_path = '_background_noise_'

    def __init__(self, root_path: str, data_tf=None, label_tf=None, include_rate=False) -> None:
        super().__init__()
        self.root_path = root_path 
        self.data_list = self.__cal_data_list__()
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.include_rate = include_rate

    def __cal_data_list__(self) -> list[str]:
        data_list = []
        for it in os.listdir(os.path.join(self.root_path, self.base_path)):
            if it.endswith('.wav'):
                data_list.append(it)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> Any:
        noise_path = os.path.join(self.root_path, self.base_path, self.data_list[index])
        noise, sample_rate = torchaudio.load(noise_path)
        noise_type = self.data_list[index][:-(len('.wav'))]
        if self.data_tf is not None:
            noise = self.data_tf(noise)
        if self.label_tf is not None:
            sample_rate = self.label_tf(sample_rate)
        if self.include_rate:
            return noise, noise_type, sample_rate
        else: return noise, noise_type

class VocalSound(Dataset):
    @dataclass
    class LabelMeta:
        index: int
        mid_name: str
        display_name: str
    @dataclass
    class AudioMeta:
        mid_name: str
        file_path: str
    def __init__(
            self, root_path: str, mode: str, include_rate=True, data_tf:torch.nn.Module=None, label_tf:torch.nn.Module=None,
            version:str='44k'
    ):
        super(VocalSound, self).__init__()
        assert mode in ['train', 'test', 'validation']
        assert version in ['44k', '16k']
        self.root_path = root_path
        self.include_rate = include_rate
        self.data_tf = data_tf 
        self.label_tf = label_tf
        self.data_path = os.path.join(root_path, 'data_44k' if version == '44k' else 'audio_16k')
        self.version = version
        
        self.label_dict = self.__label_dict__()
        self.sample_list = self.__file_list__(mode=mode)

    def __label_dict__(self, label_dic_file='class_labels_indices_vs.csv') -> dict[str, LabelMeta]:
        label_indices = pd.read_csv(os.path.join(self.root_path, label_dic_file))
        ret = {}
        for row_id, row in label_indices.iterrows():
            meta = self.LabelMeta(
                index=row['index'], mid_name=row['mid'], display_name=row['display_name']
            )
            ret[row['mid']] = meta
        return ret
    
    def __file_list__(self, mode:str) -> list[AudioMeta]:
        import json
        if mode == 'train':
            config_file_name = 'tr_rev.json' if self.version == '44k' else 'tr.json'
        elif mode == 'validation':
            config_file_name = 'val_rev.json' if self.version == '44k' else 'val.json'
        elif mode == 'test':
            config_file_name = 'te_rev.json' if self.version == '44k' else 'te.json'
        else:
            raise Exception('No support')
        with open(os.path.join(self.root_path, 'datafiles', config_file_name), 'r') as f:
            json_str = json.load(f)
        cfg_infos = pd.json_normalize(json_str['data'])
        ret = []
        for row_id, row in cfg_infos.iterrows():
            meta = self.AudioMeta(
                mid_name=row['labels'], file_path=os.path.join(self.data_path, str(row['wav']).split('/')[-1])
            )
            ret.append(meta)
        return ret

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        audioMeta = self.sample_list[index]
        label = self.label_dict[audioMeta.mid_name].index
        wavform, sample_rate = torchaudio.load(audioMeta.file_path)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        
        if self.include_rate:
            return wavform, label, sample_rate
        else:
            return wavform, label
        
class SpeechCommandsV1(Dataset):
    test_meta_file = 'testing_list.txt'
    val_meta_file = 'validation_list.txt'
    label_dic = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9., 'bed': 10., 'dog': 11., 'happy': 12., 'marvin': 13., 'off': 14., 
        'right': 15., 'up': 16., 'yes': 17., 'bird': 18., 'down': 19., 'house': 20., 'on': 21., 
        'stop': 22., 'tree': 23., 'cat': 24., 'go': 25., 'left': 26., 'no': 27., 'sheila': 28., 
        'wow': 29.
    }

    def __init__(self, root_path: str, mode: str, include_rate=True, data_tfs=None) -> None:
        super().__init__()
        self.root_path = root_path
        assert mode in ['train', 'validation', 'test', 'full', 'test+val'], 'mode type is incorrect'
        self.mode = mode
        self.include_rate = include_rate
        self.data_list = self.__cal_data_list__(mode=mode)
        self.data_tfs = data_tfs

    def __cal_data_list__(self, mode: str) -> list[str]:
        if mode == 'validation':
            with open(os.path.join(self.root_path, self.val_meta_file), 'rt', newline='\n') as f:
                val_meta_data = f.readlines()
            return [line.rstrip('\n') for line in val_meta_data]
        elif mode == 'test':
            with open(os.path.join(self.root_path, self.test_meta_file), 'rt', newline='\n') as f:
                test_meta_data = f.readlines()
            return [line.rstrip('\n') for line in test_meta_data]
        elif mode == 'full':
            full_meta_data = []
            for k,v in self.label_dic.items():
                base_path = os.path.join(self.root_path, k)
                for p in os.listdir(base_path):
                    if p.endswith('.wav'):
                        full_meta_data.append(f'{k}/{p}')
            return full_meta_data
        elif mode == 'test+val':
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            for it in val_meta_data:
                test_meta_data.append(it)
            return test_meta_data
        else:
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            full_meta_data = self.__cal_data_list__(mode='full')
            train_meta_data = []
            for it in full_meta_data:
                if it in val_meta_data:
                    continue
                if it in test_meta_data:
                    continue
                train_meta_data.append(it)
            return train_meta_data

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> torch.Tensor:
        audio_path, label = self.__cal_audio_path_label__(self.data_list[index])
        audio, sample_rate = torchaudio.load(audio_path)
        if self.data_tfs is not None:
            audio = self.data_tfs(audio)
        if self.include_rate:
            return audio, label, sample_rate
        return audio, int(label)

    def __cal_audio_path_label__(self, meta_data: str) -> tuple[str, float]:
        label = meta_data.strip().split('/')[0]
        label = self.label_dic[label]
        audio_path = os.path.join(self.root_path, meta_data)
        return audio_path, label
    
class AudioMNIST(Dataset):
    def __init__(self, data_paths: list[str], data_trainsforms=None, include_rate=True):
        super(AudioMNIST, self).__init__()
        self.data_paths = data_paths
        self.data_trainsforms = data_trainsforms
        self.include_rate = include_rate

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index) -> tuple[object, float]:
        (wavform, sample_rate) = torchaudio.load(self.data_paths[index])
        label = self.data_paths[index].split('/')[-1].split('_')[0]
        if self.data_trainsforms is not None:
            wavform = self.data_trainsforms(wavform)
        if self.include_rate:
            return (wavform, sample_rate), int(label)
        else:
            return wavform, int(label)
    
    @staticmethod
    def default_splits(mode:str, root_path:str, fold:int=0) -> list[str]:
        """ This is the original paper's splits
        "   see: https://github.com/soerenab/AudioMNIST/blob/master/preprocess_data.py
        """
        assert mode in ['train', 'test', 'validate'], 'No support'
        assert fold in [0, 1, 2, 3, 4], 'No support'
        splits = {
            'train': [
                set([
                    28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,
                    8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55
                ]),
                set([
                    36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3,
                    10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50
                ]),
                set([
                    43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, 
                    4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51
                ]),
                set([
                    12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,
                    5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53
                ]),
                set([
                    26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,
                    6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54
                ])
            ], 
            'test': [
                set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])
            ],
            "validate":[
                set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])
            ]
        }

        target_split = splits[mode][fold]

        dataset_list = []
        for it in target_split:
            data_path = os.path.join(root_path, f'{it}' if it > 9 else f'0{it}')
            for f in os.listdir(data_path):
                if f.endswith('wav'):
                    dataset_list.append(os.path.join(data_path, f))
        return dataset_list