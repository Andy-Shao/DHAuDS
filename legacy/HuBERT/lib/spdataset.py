import os

from torch.utils.data import Dataset
from torch import nn
import torchaudio

class SpeechCommandsV2(Dataset):
    test_list_file = 'testing_list.txt'
    val_list_file = 'validation_list.txt'
    label_dic = {
        'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven': 7,
        'eight':8, 'nine':9, 'bed':10, 'down':11, 'forward':12, 'house':13, 'tree':14, 'visual':15,
        'bird':16, 'learn':17, 'no':18, 'stop':19, 'wow':20, 'cat':21, 'go':22, 'left':23, 
        'off':24, 'up':25, 'yes':26, 'backward':27, 'dog':28, 'follow':29, 'happy':30, 'marvin': 31,
        'on':32, 'sheila': 33, 'three':34
    }
    def __init__(self, root_path:str, mode:str, include_rate:bool=False, data_tf:nn.Module=None, label_tf:nn.Module=None):
        super(SpeechCommandsV2, self).__init__()
        assert mode in ['train', 'test', 'val'], 'No support'
        self.root_path = root_path
        self.mode = mode
        self.include_rate = include_rate
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.data_list = self.__cal_list__(mode=mode)

    def __scanner_(self) -> list[str]:
        ret = []
        for k,v in self.label_dic.items():
            for path in os.listdir(os.path.join(self.root_path, k)):
                if str(path).endswith('.wav'):
                    ret.append(os.path.join(k, path))
        return ret
    
    def __read_list_file__(self, mode:str) -> list[str]:
        assert mode in ['test', 'val'], 'No support'
        file_path = os.path.join(self.root_path, self.test_list_file if mode == 'test' else self.val_list_file)
        with open(file=file_path, mode='r') as f:
            lists = f.readlines()
        return [it.strip() for it in lists]
    
    def __cal_list__(self, mode:str) -> list[str]:
        if mode == 'test':
            return self.__read_list_file__(mode=mode)
        elif mode == 'val':
            return self.__read_list_file__(mode=mode)
        elif mode == 'train':
            test_list = self.__cal_list__(mode='test')
            val_list = self.__cal_list__(mode='val')
            full_list = self.__scanner_()
            train_list = list(filter(lambda x: x not in test_list and x not in val_list, full_list))
            return train_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_path = self.data_list[index]
        label = int(self.label_dic[file_path.split('/')[0]])
        full_path = os.path.join(self.root_path, file_path)
        wavform, sample_rate = torchaudio.load(full_path, normalize=True)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        if self.include_rate:
            return wavform, label, sample_rate
        else: return wavform, label