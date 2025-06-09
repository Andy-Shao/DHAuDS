# NTTA

## Software Environment
```shell
conda create --name my-audio python=3.12 -y
conda activate my-audio
# CUDA 12.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
#conda install -c anaconda scipy==1.15.3 -y
conda install pandas==2.2.3 -y
conda install tqdm==4.67.1 -y
conda install matplotlib==3.10.0 -y
conda install jupyter==1.1.1 -y
pip install wandb==0.19.11
```

## Datasets
### SpeechCommands V2
The SpeechCommands V2 (2.26GB) is a speech audio set which includes 35 English words. 
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

### CochlScene
Cochl Acoustic Scene Dataset, or CochlScene, is a new acoustic scene dataset whose recordings are fully collected from crowdsourcing participants. Most of the initial plans and guidelines for the processes were provided by the researchers in the field of audio signal processing and machine learning including the authors, and the actual process was performed by using the crowdsourcing platform developed by SelectStar, a Korean crowdsourcing data company. During the process, the initial plans were reinforced and modified from the discussion about the actual difficulty in the collection process. After extracting the subset of the total collections considering the purpose of the data, we collected 76,115 10 seconds files in 13 different acoustic scenes from 831 participants.

+ Sample rate: 44100
+ Sample size: 76115 (Train: 60855, validation: 7573, test: 7687)
+ Audio length: 10 seconds
+ Class Number: 13

[Github Link](https://github.com/cochlearai/cochlscene)<br/>
[Dataset Link](https://zenodo.org/records/7080122)

## Loss Function
### Nuclear-norm Maximization Loss
$$
\begin{align}
    \mathcal{L}_{nuc-max} = - ||A||_{F} \\
    ||A||_{F} = \sqrt{\sum_i^m \sum_j^n |A_{ij}|^2}
\end{align}
$$
## Generalized Entropy Loss
$$
\begin{equation}
    \mathcal{L}_{G-entropy} = \frac{1-\sum_{i=1}^C \hat{p}_i^q}{q-1}
\end{equation}
$$
where:
+ Shannon entropy (usual cross-entropy) when $q \rightarrow 1$
+ More robust to noisy labels when $q < 1$
+ More sensitive to confident predictions when $q > 1$
+ $C$ is the number of classes
+ $y_i \in [0,1]$ is the true label (one-hot encoded)
+ $\hat{p}_i \in [0,1]$ is the predicted probability for class $i$
## Entropy Loss
$$
\begin{equation}
    \mathcal{L}_{entropy} = -\sum_{i=1}^C y_i \log(\hat{p}_i)
\end{equation}
$$

## Code Reference
[MMAuT](https://github.com/Andy-Shao/MMAuT)
