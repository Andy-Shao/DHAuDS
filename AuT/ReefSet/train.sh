#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.ReefSet.train --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --background_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 33 --lr '1e-3' --num_workers 16 --wandb