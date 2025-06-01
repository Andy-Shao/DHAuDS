#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/vocalsound_16k' --background_type 'VocalSound' --corruption_level 3.0 \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-cls0.pt'