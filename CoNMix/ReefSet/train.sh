#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.ReefSet.train --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --max_epoch 15 --interval 15 --batch_size 63 --trte full --normalized --num_workers 16 --wandb