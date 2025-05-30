import os
from typing import Any

from torch.utils.data import Dataset
import torch 
import torchaudio

class SpeechCommandsV2(Dataset):
    label_dict = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9., 'bed': 10., 'dog': 11., 'happy': 12., 'marvin': 13., 'off': 14., 
        'right': 15., 'up': 16., 'yes': 17., 'bird': 18., 'down': 19., 'house': 20., 'on': 21., 
        'stop': 22., 'tree': 23., 'cat': 24., 'go': 25., 'left': 26., 'no': 27., 'sheila': 28., 
        'wow': 29., 'backward': 30., 'forward': 31., 'follow': 32., 'learn': 33., 'visual': 34.
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

class BackgroundNoiseDataset(Dataset):
    base_path = '_background_noise_'

    def __init__(self, root_path: str, data_tf=None, label_tf=None) -> None:
        super().__init__()
        self.root_path = root_path 
        self.data_list = self.__cal_data_list__()
        self.data_tf = data_tf
        self.label_tf = label_tf

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
        return noise_type, noise, sample_rate