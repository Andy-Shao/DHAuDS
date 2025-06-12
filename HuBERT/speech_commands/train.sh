#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.speech_commands.train --dataset 'SpeechCommandsV1'  --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --batch_size 32 --max_epoch 30 --lr '1e-3' --model_level 'base' --lr_cardinality 50 \
    --use_pre_trained_weigth --wandb
