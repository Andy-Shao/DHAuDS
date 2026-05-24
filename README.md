# DHAuDS: A Dynamic and Heterogeneous Audio Benchmark for Test-Time Adaptation

## Software Environment

+ Docker image: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
+ GPU: RTX 5090 / A100 SXM4 80GB / RTX 4090 / RTX PRO 6000 WS
```shell
conda create --name DHAuDS python==3.13.9 -y
conda activate DHAuDS
# CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# pip install transformers==4.53.3
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas==2.3.1
pip install matplotlib==3.10.3
pip install jupyter==1.1.1
pip install soundfile==0.13.1
pip install wandb==0.21.0
```
<!-- ### For CoNMix++
#### Additional Installment
```shell
pip install ml-collections==1.1.0
```
#### Pre-trained weight
```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
mkdir -p model/vit_checkpoint/imagenet21k
mv R50+ViT-B_16.npz model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
``` -->

## Processing
```shell
export BASE_PATH=${the parent directory of the project}
git clone https://github.com/Andy-Shao/DHAuDS.git
conda activate DHAuDS
cd DHAuDS
```
### Training
```shell
sh HuBERT/SpeechCommandsV2/train.sh
```
### Adaptation
You should run all experiment of <code>tta.sh</code>.
```shell
sh HuBERT/SpeechCommandsV2/tta.sh
```
### Analysis
[Trained weight (tar.gz file)](https://drive.google.com/file/d/1Q-GA1CtpdEu-8S_pFP0cqMKzHI9OpvpY/view)
```shell
sh HuBERT/SpeechCommandsV2/analysis.sh
```

## Corruption Example
This chapter presents an example of corrupting SpeechCommandsV2. You can see more details from [corruption_example.ipynb](https://github.com/Andy-Shao/DHAuDS/blob/main/corruption_example.ipynb)
### WHN
```python
from lib.corruption import WHN
from lib.spdataset import SpeechCommandsV2

sc2_set = SpeechCommandsV2(
    root_path='/root/data/', mode='testing', download=True,
    data_tf=WHN(lsnr=6, rsnr=7, step=.5)
)
```
### PSH
```python
from lib.corruption import DynPSH
from lib.dataset import GpuMultiTFDataset

# Without the GPU version:
# sc2_set = SpeechCommandsV2(
#     root_path='/root/data', mode='testing', download=True,
#     data_tf=DynPSH(sample_rate=16000, min_steps=4, max_steps=5, is_bidirection=True)
# )

# Using GPU to speed up the PSH processing:
sc2_set = GpuMultiTFDataset(
    dataset=SpeechCommandsV2(
        root_path='/root/data', mode='testing', download=True
    ), device='cuda', maintain_cpu=True,
    tfs=[
        DynPSH(sample_rate=16000, min_steps=4, max_steps=5, is_bidirection=True)
    ]
)
```
### TST
```python
from lib.corruption import DynTST
from torch.utils.data import DataLoader
from lib.component import Components, AudioPadding

# One thread version:
# sc2_set = SpeechCommandsV2(
#     root_path='/root/data', mode='testing', download=True,
#     data_tf=DynTST(min_rate=.04, max_rate=.06, step=.01)
# )

# Using multiple CPU cores to speed up
## Note: batch process needs sample audio length, so we pad samples
sc2_set = SpeechCommandsV2(
    root_path='/root/data', mode='testing', download=True,
    data_tf=Components(transforms=[
        DynTST(min_rate=.04, max_rate=.06, step=.01),
        AudioPadding(max_length=16000, sample_rate=16000, random_shift=False)
    ])
)
sc2_loader = DataLoader(
    dataset=sc2_set, batch_size=32, shuffle=False, drop_last=False,
    num_workers=16
)
```
### ENSC
```python
from lib.corruption import DynEN, ensc_noises

# Read ENSC noises
noise_list = ensc_noises(
    ensc_path='/root/data', noise_modes=['exercise_bike', 'running_tap', 'white_noise', 'pink_noise'], 
    sample_rate=16000
)

# Corrupting
sc2_set = SpeechCommandsV2(
    root_path='/root/data', mode='testing', download=True,
    data_tf=DynEN(noise_list=noise_list, lsnr=5, rsnr=6, step=.5)
)
```
### ENQ
```python
from lib.corruption import enq_noises

# You need to download the QUT-NOISE dataset by yourself
# Read QUT-NOISE noises
noise_list = enq_noises(
    enq_path='/root/data/QUT-NOISE', noise_modes=['HOME', 'REVERB', 'STREET'], sample_rate=16000
)

# Corrupting
sc2_set = SpeechCommandsV2(
    root_path='/root/data', mode='testing', download=True,
    data_tf=DynEN(noise_list=noise_list, lsnr=5, rsnr=6, step=.5)
)
```
### END1
```python
from lib.corruption import end_noises

# You need to download the DEMAND dataset (16 kHz version) by yourself
# Read DEMAND noises
noise_list = end_noises(
    end_path='/root/data/DEMAND_16k', noise_modes=['NFIELD', 'PRESTO', 'TCAR', 'OOFFICE'],
    sample_rate=16000
)
# Note: For DEMAND 48 kHz version, use the end_noises_48k(..) function instead of end_noses(..)

# Corrupting
sc2_set = SpeechCommandsV2(
    root_path='/root/data', mode='testing', download=True,
    data_tf=DynEN(noise_list=noise_list, lsnr=5, rsnr=6, step=.5)
)
```
### END2
```python
# You need to download the DEMAND dataset by yourself
noise_list = end_noises(
    end_path='/root/data/DEMAND_16k', noise_modes=['DLIVING', 'OHALLWAY', 'SPSQUARE', 'TMETRO'],
    sample_rate=16000
)
sc2_set = SpeechCommandsV2(
    root_path='/root/data', mode='testing', download=True,
    data_tf=DynEN(noise_list=noise_list, lsnr=5, rsnr=6, step=.5)
)
```

## Datasets
### SpeechCommands V2
The SpeechCommands V2 (2.26GB) is a speech audio set that includes 35 English words. 
+ Sample size: 105,829 (train: 84,843, test: 11,005, validation: 9,981)
+ Sampling rate: 16 kHz
+ Class Number: 35
+ One sample length: 1s
  
[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Pytorch Document](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html)

### SpeechCommands V2-C
SpeechCommandsV2-C (SC2-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
SC2-C, a subset of [SpeechCommands V2](https://research.google/blog/launching-the-speech-commands-dataset), utilizes only test-set samples to address domain shift. 
Each sample in SC2-C has a duration of 1 second, a sample rate of 16 kHz, and belongs to one of 35 classes. The dataset comprises two sets: adaptation and 
evaluation sets. Each set comprises seven corruption categories and two levels. Seven corruption categories are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. 
Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. As for the evaluation and adaptation sets, each comprises 154,070 samples. 
In total, SC2-C consists of 308,140 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 11,005 samples.
+ Sample size: 308,140 (11,005 per category-level)
+ Sample rate: 16 kHz
+ Class number: 35
+ One sample length: 1s

[SC2-C Dataset Link](https://drive.google.com/drive/folders/1wBCadjjcA-n7fCAvf82uYBR6q5z_uXRm)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/SpeechCommandsV2-C)

### VocalSound
VocalSound is a free dataset consisting of 21,024 crowdsourced recordings of laughter, sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects. The VocalSound dataset also contains meta-information such as speaker age, gender, native language, country, and health condition.
+ Sample size: 20,977 (Train: 15,531, validation: 1,855, test: 3,591)
+ Sample rate: 16 kHz
+ One sample length: less than 12s
+ Class number: 6

[VocalSound Dataset Link](https://sls.csail.mit.edu/downloads/vocalsound/)<br/>
Download command:
```shell
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

### VocalSound-C
VocalSound-C (VS-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
VS-C, a subset of [VocalSound](https://sls.csail.mit.edu/downloads/vocalsound), utilizes only 
test-set samples to address domain shift. Each sample in VS-C has a duration of 10 seconds, a sample rate of 16 kHz, and belongs to one of 6 classes. 
The dataset comprises two sets: adaptation and evaluation sets. Each set comprises seven corruption categories and two levels. Seven corruption categories 
are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. As for evaluation 
and adaptation sets, each comprises 50,274 samples. In total, VS-C consists of 100,548 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 
3,591 samples.
+ Sample size: 100,548 (3,591 per category-level)
+ Sample rate: 16 kHz
+ One sample length: 10s
+ Class number: 6

[VS-C Dataset Link](https://drive.google.com/drive/folders/1QysFmdmFUQgQ0BlADU4eJ_xuHUxziSJX)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/VocalSound-C)

### UrbanSound8K
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy.

+ Sample size: 8,732
+ Sample rate: less than 192000 (different audio, different sample rate)
+ One sample length: less than 4s
+ Class number: 10

Dataset download:
```shell
pip install soundata
```
```python
import soundata

# learn wich datasets are available in soundata
print(soundata.list_datasets())

# choose a dataset and download it
dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live')
dataset.download()

# get annotations and audio for a random clip
example_clip = dataset.choice_clip()
tags = example_clip.tags
y, sr = example_clip.audio
```

### UrbanSound8K-C
UrbanSound8K-C (US8-C) serves as a benchmark for DHAuDS in the context of 
test-time adaptation for audio classification. US8-C, a subset of [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), 
utilizes only test-set samples to address domain shift. Each sample in US8-C has a duration of 4 seconds, a sample rate of 44.1 kHz, 
and belongs to one of 10 classes. The dataset comprises two sets: adaptation and evaluation sets. Each set comprises four corruption categories and two levels. 
Four corruption categories are defined: WHN, ENSC, TST, and PSH. Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. 
As for evaluation and adaptation sets, each comprises 9,836 samples. In total, US8-C consists of 19,672 samples, with each category-level, such as WHN-L1 or 
WHN-L2, containing 2,459 samples.
+ Sample size: 19,672 (2,459 per category-level)
+ Sample rate: 44.1 kHz
+ One sample length: 4s
+ Class Number: 10

[US8K-C Dataset Link](https://drive.google.com/drive/folders/1kUzBwwrRO5sIq8GUhGf8FP4HbCnb7KTh)<br/>
[Hugging face backup](https://huggingface.co/datasets/AndyShao90/UrbanSound8K-C)

### ReefSet
ReefSet is a multi-labeled and imbalanced dataset. ReefSet compiled a diverse meta-dataset of 57084 labelled coral reef bioacoustic recordings across 37 classes and from 16 individual datasets over 12 countries. During the annotation of each dataset, longer recording periods were segmented into samples of shorter windows (1.88 s) to fit within the two window lengths of the industry-standard networks. The final meta-dataset of 57074 labelled samples, split across the four primary labels: biophony (79.20%), anthrophony (10.39%), geophony (0.09%), and ambient (10.32%), with 33 secondary labels.

+ Sample rate: 16 kHz
+ Sample size: 57,074
+ One sample length: 1.88s
+ Class Number: 37 (4 primary labels and 33 secondary labels)

[Official Link](https://zenodo.org/records/11071202)

### ReefSet-C
ReefSet-C (RS-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
RS-C, a subset of [ReefSet](https://zenodo.org/records/11071202), utilizes only test-set samples to address domain shift. Each sample in RS-C has a duration 
of 1.88 seconds, a sample rate of 16 kHz, and belongs to one of 37 classes. The dataset comprises two sets: adaptation and evaluation sets. Each set comprises 
seven corruption categories and two levels. Seven corruption categories are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. Two levels are defined: L1 and 
L2, where L2 indicates a higher degree of complexity. As for evaluation and adaptation sets, each comprises 239,918 samples. In total, RS-C consists of 
479,836 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 17,137 samples.
+ Sample size: 479,836 (17,137 per category-level)
+ Sample rate: 16 kHz
+ One sample length: 1.88s

[RS-C Dataset Link](https://drive.google.com/drive/folders/1W9GGOZTq3XSSsOlpJOQueDksHkCn3Fj4)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/ReefSet-C)

### QUT-NOISE
QUT-NOISE is an environmental acoustic dataset for environmental background noise. QUT-NOISE comprises five distinct types of background noise: CAFE, CAR, HOME, REVERB, and STREET. Each type of noise includes five noise files.
+ Sample rate: 48 kHz
+ Sample size: 20
+ One sample length: greater than 1990s (33m 10s)
+ Class Number: 5
  
[Official Link](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/)

### DEMAND
A database of 16-channel environmental noise recordings. 
+ Sample rate: 16 kHz/48 kHz 
+ Sample size: 272
+ One sample length: 300s
+ Class Number: 16
  
[Official Link](https://zenodo.org/records/1227121)

## Code Reference
+ [AMAuT](https://github.com/Andy-Shao/AMAuT)
+ [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
+ [TTA in Audio Classification](https://github.com/Andy-Shao/TTA-in-AC.git)

## Citation
```text
@article{shao2025dhauds,
  title={DHAuDS: A Dynamic and Heterogeneous Audio Benchmark for Test-Time Adaptation},
  author={Shao, Weichuang and Liao, Iman Yi and Maul, Tomas Henrique Bode and Chandesa, Tissa},
  journal={arXiv preprint arXiv:2511.18421},
  year={2025}
}
```
