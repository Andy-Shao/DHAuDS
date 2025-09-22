#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m Hyb.SpeechCommandsV2.analysis --dataset SpeechCommandsV2 --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'SC2_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --batch_size 32 \
    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/TTDA' \
    --hub_wght_path './result/SpeechCommandsV2/HuBERT/TTDA'