#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/vocalsound_16k' --background_type 'VocalSound' --corruption_level 10.0 \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/HuBERT/train/hubert-base.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/HuBERT/train/clsModel-base.pt'

python -m HuBERT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/CochlScene' --background_type 'CochlScene' \
    --corruption_level 10.0 --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/HuBERT/train/hubert-base.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/HuBERT/train/clsModel-base.pt'