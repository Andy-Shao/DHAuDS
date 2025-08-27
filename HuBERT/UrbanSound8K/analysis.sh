#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Analysis data
python -m HuBERT.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 33 --output_file_name 'UrbanSound8K_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/UrbanSound8K/HuBERT/train' \
    --adpt_wght_path './result/UrbanSound8K/HuBERT/TTDA'