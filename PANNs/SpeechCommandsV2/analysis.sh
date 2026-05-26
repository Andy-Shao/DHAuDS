#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --output_file_name 'PAN_SC2-C_analysis.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/PANNs/TTDA'