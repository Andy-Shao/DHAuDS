#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.speech_commands.ttda2 --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --noise_path $BASE_PATH'/data/QUT-NOISE'