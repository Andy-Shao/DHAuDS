#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.SpeechCommandsV2.train --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
     --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --wandb