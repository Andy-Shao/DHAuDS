#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.speech_commands.train --dataset 'SpeechCommandsV1'  --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --batch_size 32 --max_epoch 15 --lr '8e-4' --lr_cardinality 50 --lr_gamma 30 --wandb