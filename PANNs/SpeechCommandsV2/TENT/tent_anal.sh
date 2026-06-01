#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-5 \
    --output_file_name 'PAN_SC2-C_TENT_analysis.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train'