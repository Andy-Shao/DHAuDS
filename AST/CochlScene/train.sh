#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.CochlScene.train --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --batch_size 30 --max_epoch 10 --lr '8e-4' --lr_gamma 30 --wandb