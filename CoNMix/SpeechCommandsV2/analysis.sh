#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --output_file_name 'SpeechCommandsV2_analysis.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/CoNMix/train' \
    --adpt_wght_path './result/SpeechCommandsV2/CoNMix/STDA'
