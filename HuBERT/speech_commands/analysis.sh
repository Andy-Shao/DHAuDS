#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.speech_commands.analysis --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' --cache_path $BASE_PATH'/tmp' \
    --corruption_level 3.0 --analysis_file 'analysis.3.0.csv' --noise_types 'doing_the_dishes' \
    --softmax --batch_size 32 --softmax \
    --origin_auT_weight './result/SpeechCommandsV1/HuBERT/train/hubert-base-SC1.pt' \
    --origin_cls_weight './result/SpeechCommandsV1/HuBERT/train/clsModel-base-SC1.pt' \
    --adapted_auT_weight './result/SpeechCommandsV1/HuBERT/TTA/HuBERT-SC1-hubert-DTD-3.0.pt' \
    --adapted_cls_weight './result/SpeechCommandsV1/HuBERT/TTA/HuBERT-SC1-clsModel-DTD-3.0.pt'