#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/speech_commands_v0.02/speech_commands_v0.02' \
    --vocalsound_path $BASE_PATH'/data/vocalsound_16k' \
    --cochlscene_path $BASE_PATH'/data/CochlScene' \
    --corruption_level 10.0 --analysis_file 'analysis.10.0.csv' \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-cls0.pt'

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/speech_commands_v0.02/speech_commands_v0.02' \
    --vocalsound_path $BASE_PATH'/data/vocalsound_16k' \
    --cochlscene_path $BASE_PATH'/data/CochlScene' \
    --corruption_level 3.0 --analysis_file 'analysis.3.0.csv' \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-cls0.pt'