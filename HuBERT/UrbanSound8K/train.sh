#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.UrbanSound8K.train --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --batch_size 33 --lr '5e-4' --max_epoch 60 --model_level 'base' --use_pre_trained_weigth \
    --lr_cardinality 50 --lr_gamma 30 --lr_threshold 20 --wandb