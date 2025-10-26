#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.SpeechCommandsV2.train --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --max_epoch 20 --interval 20 --batch_size 64 --trte full --normalized --num_workers 16 --wandb