import argparse

from torch.utils.data import Dataset
from torch import nn
import torch

def dataset_tag(dataset:str) -> str:
    if dataset == 'SpeechCommandsV2':
        return 'SC2'
    else:
        raise Exception('No support')

class TransferDataset(Dataset):
    def __init__(self, dataset: Dataset, data_tf:nn.Module=None, label_tf:nn.Module=None, device='cpu', keep_cpu=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_tf = data_tf if device == 'cpu' or data_tf is None else data_tf.to(device=device)
        self.label_tf = label_tf if device == 'cpu' or label_tf is None else label_tf.to(device=device)
        self.device = device
        self.keep_cpu = keep_cpu

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        feature, label = self.dataset[index]
        if self.device != 'cpu':
            feature = feature.to(self.device)
            label = label.to(self.device) if isinstance(label, torch.Tensor) else label
        if self.data_tf is not None:
            feature = self.data_tf(feature)
        if self.label_tf is not None:
            label = self.label_tf(label)

        if self.device != 'cpu' and self.keep_cpu:
            return feature.cpu(), label.cpu() if isinstance(label, torch.Tensor) else label
        else:
            return feature, label

class MultiTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tfs:list[nn.Module]):
        super(MultiTFDataset, self).__init__()
        assert tfs is not None, 'No support'
        self.dataset = dataset
        self.tfs = tfs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        ret = [item.clone() for _ in range(len(self.tfs))]
        for i, tf in enumerate(self.tfs):
            if tf is not None:
                ret[i] = tf(ret[i])
        ret.append(label)
        return tuple(ret)

class RandomChoiceSet(Dataset):
    def __init__(self, dataset:Dataset, include_lables:list=[]):
        super(RandomChoiceSet, self).__init__()
        self.dataset = dataset
        self.include_labels = include_lables
        self.data_list = self.__cal_data_list__()

    def __cal_data_list__(self) -> list[int]:
        if len(self.include_labels) == 0:
            return [i for i in range(len(self.dataset))]
        else:
            return [i for i, (_, label) in enumerate(self.dataset) if label in self.include_labels]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        import random
        i = random.randint(0, len(self.data_list)-1)
        return self.dataset[self.data_list[i]]