#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.VocalSound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --background_path $BASE_PATH \
    --max_epoch 35 --batch_size 32 --lr '1e-3' --lr_cardinality 80 --wandb