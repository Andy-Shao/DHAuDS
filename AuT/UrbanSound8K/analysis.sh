#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --output_file_name 'US8_AuT.csv' \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train'