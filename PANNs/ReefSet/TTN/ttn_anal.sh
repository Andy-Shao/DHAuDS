#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.ReefSet.TTN.ttn_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_TTN_analysis.csv' --batch_size 32 \
    --lr 1e-5 --adpt_batch_size 70 \
    --orig_wght_pth './result/ReefSet/PANNs/train'