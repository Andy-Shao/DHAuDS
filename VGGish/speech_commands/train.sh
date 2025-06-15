#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m VGGish.speech_commands.train --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
     --batch_size 32 --max_epoch 80 --lr '1e-3' --lr_cardinality 50 --lr_gamma 10 --lr_threshold 20 --wandb