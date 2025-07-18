#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.AudioMNIST.train --dataset 'AudioMINST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --max_epoch 10 --lr '1e-4' --lr_gamma 30 --wandb