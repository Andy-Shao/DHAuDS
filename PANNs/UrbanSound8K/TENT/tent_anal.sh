#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_US8-C_TENT_analysis.csv' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train'