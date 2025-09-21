# A Dynamic and Heterogeneous Audio Domain Shift (DHAuDS) Benchmark for Test-time Adaptation on Audio Classification

## Software Environment

+ Docker image: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
+ GPU: RTX 5090
```shell
conda create --name DHAuDS python=3.13 -y
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

## Datasets
### SpeechCommands V2
The SpeechCommands V2 (2.26GB) is a speech audio set that includes 35 English words. 
+ Sample size: 105829 (train: 84843, test: 11005, validation: 9981)
+ Sampling rate: 16 kHz
+ Class Number: 35
+ One sample length: 1 second
  
[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Pytorch Document](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html)

### VocalSound
VocalSound is a free dataset consisting of 21,024 crowdsourced recordings of laughter, sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects. The VocalSound dataset also contains meta-information such as speaker age, gender, native language, country, and health condition.
+ Sample size: 20977 (Train: 15531, validation: 1855, test: 3591)
+ Sample rate: 16 kHz
+ One sample length: less than 12 seconds
+ Class number: 6

[VocalSound Dataset Link](https://sls.csail.mit.edu/downloads/vocalsound/)<br/>
Download command:
```shell
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

### UrbanSound8K
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy.

+ Sample size: 8732
+ Sample rate: less than 192000 (different audio different sample rate)
+ One sample length: less than 4 seconds
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

### ReefSet
ReefSet is a multi-labeled and imbalanced dataset. ReefSet compiled a diverse meta-dataset of 57084 labelled coral reef bioacoustic recordings across 37 classes and from 16 individual datasets over 12 countries. During the annotation of each dataset, longer recording periods were segmented into samples of shorter windows (1.88 s) to fit within the two window lengths of the industry-standard networks. The final meta-dataset of 57074 labelled samples, split across the four primary labels: biophony (79.20%), anthrophony (10.39%), geophony (0.09%), and ambient (10.32%), with 33 secondary labels.

+ Sample rate: 16 kHz
+ Sample size: 57074
+ One sample length: 1.88s
+ Class Number: 37 (4 primary labels and 33 secondary labels)

[Official Link](https://zenodo.org/records/11071202)

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
