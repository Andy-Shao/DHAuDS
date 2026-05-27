#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'PAN_US8-C_analysis.csv' --batch_size 32 \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' \
    --adpt_wght_pth './result/UrbanSound8K/PANNs/TTDA'