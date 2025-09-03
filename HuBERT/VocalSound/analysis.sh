#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

## Analysis
python -m HuBERT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --output_file_name 'VocalSound_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/VocalSound/HuBERT/train' \
    --adpt_wght_path './result/VocalSound/HuBERT/TTDA'