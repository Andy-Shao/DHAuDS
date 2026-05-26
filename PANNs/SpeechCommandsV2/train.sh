#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.SpeechCommandsV2.train --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --lr '1e-3' --max_epoch 40 --wandb