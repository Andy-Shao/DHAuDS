#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.CochlScene.train --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --batch_size 30 --max_epoch 30 --lr '1e-3' --lr_gamma 30 --wandb