#!bin/bash
export BASE_PATH='/root'

# Analysis
python -m HuBERT.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'SpeechCommandsV2_analysis.csv' --batch_size 32 --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path './result/SpeechCommandsV2/HuBERT/TTDA'