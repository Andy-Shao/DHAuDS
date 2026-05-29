#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_analysis.csv' --batch_size 32 \
    --orig_wght_pth './result/ReefSet/PANNs/train' \
    --adpt_wght_pth './result/ReefSet/PANNs/TTDA'