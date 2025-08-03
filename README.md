# A Dynamic and Heterogeneous Audio Domain Shift (DHAuDS) for Test-time Adaptation on Audio Classification

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
pip install wandb==0.21.0
```
In some cloud platforms, such as [Google Cloud](https://cloud.google.com). You should install more:
```shell
pip install soundfile
```

## Datasets
### AudioMNIST
This repository contains code and data used in the Interpretation and Explanation of Deep Neural Networks for Classifying Audio Signals. The dataset consists of 30,000 audio samples of spoken digits (0–9) from 60 different speakers. Additionally, it contains the audioMNIST_meta.txt file, which provides meta information such as the gender or age of each speaker.

+ Sample size: 30000 (Train: 18000, Validation: 6000, Test: 6000)
+ Sample rate: 48 kHz
+ Audio length: 1 second
+ Class Number: 10
  
[Official Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)<br/>
[Dataset Hosting Link](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)
<!--[Dataset Hosting Link](https://drive.google.com/file/d/1kq5_qCKRUTHmViDIziSRKPjW4fIoyT9u/view?usp=drive_link)-->

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
+ Sample Size: 20977 (Train: 15531, validation: 1855, test: 3591)
+ Sample rate: 16 kHz
+ One sample length: less than 12 seconds
+ Class number: 6

[VocalSound Dataset Link](https://sls.csail.mit.edu/downloads/vocalsound/)<br/>
Download command:
```shell
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

### ReefSet
ReefSet is a multi-labeled and imbalanced dataset. ReefSet compiled a diverse meta-dataset of 57084 labelled coral reef bioacoustic recordings across 37 classes and from 16 individual datasets over 12 countries. During the annotation of each dataset, longer recording periods were segmented into samples of shorter windows (1.88 s) to fit within the two window lengths of the industry-standard networks. The final meta-dataset of 57074 labelled samples, split across the four primary labels: biophony (79.20%), anthrophony (10.39%), geophony (0.09%), and ambient (10.32%), with 33 secondary labels.

+ Sample rate: 16 kHz
+ Sample size: 57074
+ One sample length: 1.88s
+ Class Number: 37 (4 primary labels and 33 secondary labels)

[Official Link](https://zenodo.org/records/11071202)

<!-- ### CochlScene
Cochl Acoustic Scene Dataset, or CochlScene, is a new acoustic scene dataset whose recordings are fully collected from crowdsourcing participants. Most of the initial plans and guidelines for the processes were provided by researchers in the field of audio signal processing and machine learning, including the authors. The actual process was performed using a crowdsourcing platform developed by SelectStar, a Korean crowdsourcing data company. During the process, the initial plans were reinforced and modified based on the discussion about the actual difficulties in the collection process. After extracting a subset of the total collections based on the data's purpose, we collected 76,115 10-second files from 13 different acoustic scenes involving 831 participants.

+ Sample rate: 44100
+ Sample size: 76115 (Train: 60855, validation: 7573, test: 7687)
+ One sample length: 10 seconds
+ Class number: 13

[Github Link](https://github.com/cochlearai/cochlscene)<br/>
[Dataset Link](https://zenodo.org/records/7080122) -->

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
<!-- + [SSAST](https://github.com/YuanGongND/ssast/tree/main) -->
