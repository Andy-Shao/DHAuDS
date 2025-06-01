#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.speech_commands.train --dataset_root_path $BASE_PATH'/data' --batch_size 32 --max_epoch 50 \
    --lr '1e-3' --model_level 'base' --wandb
