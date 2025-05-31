#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-cls0.pt'