#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --cache_path $BASE_PATH'/tmp' --analysis_file 'analysis.csv' \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-cls0.pt' \
    --adapted_auT_weight './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-auT.pt' \
    --adapted_cls_weight './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-cls.pt'