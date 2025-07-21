#!bin/bash 
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.SpeechCommandsV2.train --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --lr '8e-4' --max_epoch 14 --model_level 'base' --use_pre_trained_weigth \
    --lr_cardinality 50 --lr_gamma 30 --wandb