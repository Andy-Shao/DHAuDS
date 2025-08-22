#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.UrbanSound8K.train --dataset UrbanSound8K --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --max_epoch 30 --batch_size 32 \
    --lr '1e-3' --wandb