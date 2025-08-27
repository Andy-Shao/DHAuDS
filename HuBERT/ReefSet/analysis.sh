#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

## Analysis
python -m HuBERT.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 33 --output_file_name 'ReefSet_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/ReefSet/HuBERT/train' \
    --adpt_wght_path './result/ReefSet/HuBERT/TTDA'