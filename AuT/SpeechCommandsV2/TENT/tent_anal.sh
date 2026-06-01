#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'AuT_SC2-C_TENT_analysis.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train'