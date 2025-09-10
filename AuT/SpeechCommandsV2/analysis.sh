#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train'