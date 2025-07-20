#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.VocalSound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 32 --max_epoch 30 --lr '8e-4' --model_level 'base' --lr_cardinality 50 --lr_gamma 30 \
    --use_pre_trained_weigth --wandb