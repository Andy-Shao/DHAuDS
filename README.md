# A Dynamic and Heterogeneous Audio Domain Shift (DHAuDS) for Test-time Adaptation on Audio Classification

## Software Environment
Docker image id: nvidia/cuda:12.4.0-runtime-ubuntu22.04
```shell
conda create --name DHAuDS python=3.12 -y
conda activate DHAuDS
# CUDA 12.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install conda-forge::transformers==4.52.4 -y
conda install -c conda-forge torchmetrics==1.7.4 -y
conda install matplotlib==3.10.0 -y
conda install jupyter==1.1.1 -y
pip install wandb==0.19.11
```

## Datasets
### AudioMNIST
This repository contains code and data used in the Interpretation and Explanation of Deep Neural Networks for Classifying Audio Signals. The dataset consists of 30,000 audio samples of spoken digits (0–9) from 60 different speakers. Additionally, it contains the audioMNIST_meta.txt file, which provides meta information such as the gender or age of each speaker.

+ Sample size: 30000 (Train: 18000, Validation: 6000, Test: 6000)
+ Sample rate: 48000
+ Audio length: 1 second
+ Class Number: 10
<!-- + sample data shape: [1, 14073 - 47998] -->
  
[Official Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)<br/>
[Dataset Hosting Link](https://drive.google.com/file/d/1kq5_qCKRUTHmViDIziSRKPjW4fIoyT9u/view?usp=drive_link)
<!-- ### SpeechCommands V1
The dataset (1.4 GB) comprises 65,000 one-second-long utterances of 30 short words, contributed by thousands of different people through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in the current implementation,
it is marked by the `True` value of `the "is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.

+ Sample size: 64721 (train: 51088, test: 6835, validation: 6798)
+ Sampling rate: 16000
+ Class Number: 30
+ Audio length: 1 second

[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Dataset Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) -->

### SpeechCommands V2
The SpeechCommands V2 (2.26GB) is a speech audio set that includes 35 English words. 
+ Sample size: 105829 (train: 84843, test: 11005, validation: 9981)
+ Sampling rate: 16000
+ Class Number: 35
+ Audio length: 1 second
  
[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Pytorch Document](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html)

### VocalSound
VocalSound is a free dataset consisting of 21,024 crowdsourced recordings of laughter, sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects. The VocalSound dataset also contains meta-information such as speaker age, gender, native language, country, and health condition.
+ Sample Size: 20977 (Train: 15531, validation: 1855, test: 3591)
+ Sample rate: 16000
+ Audio length: less than 12 seconds
+ Class number: 6

[VocalSound Dataset Link](https://sls.csail.mit.edu/downloads/vocalsound/)<br/>
Download command:
```shell
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

### ReefSet
This dataset contains strongly labeled audio clips from coral reef habitats, taken across 16 unique datasets from 11 countries. This dataset can be used to test transfer learning performance of audio embedding models. This folder includes: - 57,084 WAV files that make up ReefSet_v1.0. Each file is 1.88 seconds in length, sampled at 16 kHz. All files have an associated label. - reefset_annotations.json, which contains the associated label, file ID, filename, data sharer, dataset and recorder type used for each sample in ReefSet_v1.0. This information is also indicated in the filename of each file. - 'reefset_labels_by_dataset.csv' provides a table containing the counts of each label class within each dataset.

+ Sample rate: 16000
+ Sample size: ???
+ Audio length: 1.88s
+ Class Number: ???

### QUT-NOISE
QUT-NOISE is an environmental acoustic dataset for environmental background noise. QUT-NOISE comprises five distinct types of background noise: CAFE, CAR, HOME, REVERB, and STREET. Each type of noise includes five noise files.
+ Sample rate: 16000
+ Sample size: 20
+ Audio length: ???
+ Class Number: 5
  
[Official Link](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/)

### DEMAND
A database of 16-channel environmental noise recordings. 
+ Sample rate: 16000/48000
+ Sample size: ???
+ Audio length: ???
+ Class Number: 16
  
[Official Link](https://zenodo.org/records/1227121)

## Code Reference
+ [AMAuT](https://github.com/Andy-Shao/AMAuT)
+ [SSAST](https://github.com/YuanGongND/ssast/tree/main)
+ [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
