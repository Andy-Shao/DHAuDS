# NTTA

## Software Environment
```shell
conda create --name my-audio python=3.12
conda activate my-audio
# CUDA 12.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install pandas==2.2.3 -y
conda install tqdm==4.67.1 -y
conda install matplotlib==3.10.0 -y
conda install jupyter==1.1.1
pip install wandb==0.19.11
```
