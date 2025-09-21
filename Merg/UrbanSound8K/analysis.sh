#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m Merg.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'US8_analysis.csv' --batch_size 32 --use_pre_trained_weigth \
    --model_level 'base' \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/TTDA' \
    --hub_wght_path './result/UrbanSound8K/HuBERT/TTDA' 