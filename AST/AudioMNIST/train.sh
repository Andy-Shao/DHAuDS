#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.AudioMNIST.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --max_epoch 10 --lr '2e-4' --lr_gamma 30 --wandb