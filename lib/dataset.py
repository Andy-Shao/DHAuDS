import argparse
import os
import pandas as pd
from typing import Any
from tqdm import tqdm
import shutil
import numpy as np

from torch.utils.data import Dataset
from torch import nn
import torch

def dataset_tag(dataset:str) -> str:
    if dataset == 'SpeechCommandsV2':
        return 'SC2'
    elif dataset == 'SpeechCommandsV1':
        return 'SC1'
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

# class MgDataset(Dataset):
#     def __init__(self, dataset:Dataset):
#         super(MgDataset, self).__init__()
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         tuple = self.dataset[index]
#         label = tuple[-1]
#         ret = [torch.unsqueeze(input=tuple[i], dim=0) for i in range(len(tuple)-1)]
#         return torch.cat(ret, dim=0), label

# class SplitDataset(Dataset):
#     def __init__(self, dataset:Dataset, num:int, tfs:list[nn.Module]):
#         assert num > 0, 'No support'
#         self.dataset = dataset
#         self.num = num
#         self.tfs = tfs

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         fs, label = self.dataset[index]
#         if self.num == 1:
#             return fs, label
#         ret = []
#         items = torch.split(fs, 1, dim=0)
#         for i in range(self.num):
#             item = items[i]
#             item = torch.squeeze(item, dim=0)
#             item = torch.from_numpy(item.detach().numpy())
#             item = self.tfs[i](item)
#             ret.append(item)
#         ret.append(label)
#         return tuple(ret)

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

def store_to(dataset: Dataset, root_path: str, index_file_name: str, data_transf=None, label_transf=None) -> None:
    print(f'Store dataset into {root_path}, meta file is: {index_file_name}')
    data_index = pd.DataFrame(columns=['data_path', 'label'])
    try: 
        if os.path.exists(root_path): shutil.rmtree(root_path)
        os.makedirs(root_path)
    except:
        print('remove directory has an error.')
    for index, (feature, label) in tqdm(enumerate(dataset), total=len(dataset)):
        if data_transf is not None:
            feature = data_transf(feature)
        if label_transf is not None:
            label = data_transf(feature)
        data_path = f'{index}_{label}.dt'
        data_index.loc[len(data_index)] = [data_path, label]
        torch.save(feature, os.path.join(root_path, data_path))
    data_index.to_csv(os.path.join(root_path, index_file_name))

def mlt_store_to(dataset:Dataset, root_path:str, index_file_name:str, data_tfs:list[nn.Module], label_tf:nn.Module=None) -> None:
    print(f'Store dataset into {root_path}, meta file is: {index_file_name}')
    columns = []
    for i in range(len(data_tfs)):
        columns.append(f'data_path{i}')
    columns.append('label')
    data_index = pd.DataFrame(columns=columns)
    try: 
        if os.path.exists(root_path): shutil.rmtree(root_path)
        os.makedirs(root_path)
    except:
        raise Exception('remove director has an error')
    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        label = data[-1]
        if label_tf is not None:
            label = label_tf(label)
        row = []
        for i in range(len(data_tfs)):
            feature = data[i]
            if data_tfs[i] is not None:
                feature = data_tfs[i](feature)
            data_path = f'{idx}-{i}_{label}.npy'
            np.save(file=os.path.join(root_path, data_path), arr=feature.detach().numpy())
            row.append(data_path)
        row.append(label)
        data_index.loc[len(data_index)] = row
    data_index.to_csv(os.path.join(root_path, index_file_name))

def load_from(root_path: str, index_file_name: str, data_tf=None, label_tf=None) -> Dataset:
    class LoadDs(Dataset):
        def __init__(self) -> None:
            super().__init__()
            data_index = pd.read_csv(os.path.join(root_path, index_file_name))
            self.data_meta = []
            for idx in range(len(data_index)):
                self.data_meta.append([data_index['data_path'][idx], data_index['label'][idx]]) 
        
        def __len__(self):
            return len(self.data_meta)
        
        def __getitem__(self, index) -> Any:
            data_path = self.data_meta[index][0]
            feature = torch.load(os.path.join(root_path, data_path), weights_only=False)
            label = int(self.data_meta[index][1])
            if data_tf is not None:
                feature = data_tf(feature)
            if label_tf is not None:
                label = label_tf(label)
            return feature, label
    return LoadDs()

def mlt_load_from(root_path:str, index_file_name:str, data_tfs:list[nn.Module]=None, label_tf=None) -> Dataset:
    class MltLoadDs(Dataset):
        def __init__(self):
            super().__init__()
            self.data_index = pd.read_csv(os.path.join(root_path, index_file_name), index_col=0)
            self.length = self.data_index.shape[0]

        def __len__(self):
            return self.length
        
        def __getitem__(self, index):
            row = self.data_index.iloc[index]
            ret = []
            for i in range(row.shape[0] - 1):
                feature = np.load(os.path.join(root_path, row.iloc[i]))
                feature = torch.from_numpy(feature)
                if data_tfs is not None:
                    feature = data_tfs[i](feature)
                ret.append(feature)
            label = int(row['label'])
            if label_tf is not None:
                label = label_tf(label)
            ret.append(label)
            return tuple(ret)
    return MltLoadDs()

class GpuMultiTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tfs:list[nn.Module], device:str='cuda', maintain_cpu:bool=True):
        super(GpuMultiTFDataset, self).__init__()
        assert tfs is not None, 'No support'
        self.dataset = dataset
        self.tfs = [tf.to(device) for tf in tfs]
        self.device = device
        self.maintain_cpu = maintain_cpu

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        item = item.to(self.device)
        ret = [item.clone() for _ in range(len(self.tfs))]
        for i, tf in enumerate(self.tfs):
            if tf is not None:
                ret[i] = tf(ret[i])
                if self.maintain_cpu:
                    ret[i] = ret[i].to('cpu')
        ret.append(label)
        return tuple(ret)

class IdxSet(Dataset):
    def __init__(self, dataset:Dataset, idx:int=0):
        super().__init__()
        self.dataset = dataset
        self.idx = idx
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return data[self.idx], data[-1]