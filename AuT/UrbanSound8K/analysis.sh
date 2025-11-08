#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --batch_size 32 --output_file_name 'US8_AuT.csv' \
#     --orig_wght_pth './result/UrbanSound8K/AMAuT/train' \
#     --adpt_wght_path './result/UrbanSound8K/AMAuT/TTDA'

python -m AuT.UrbanSound8K.silhouette_analy --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --output_file_name 'silhouette_analysis.csv' \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' \
    --adpt_wght_path './result/UrbanSound8K/AMAuT/TTDA'