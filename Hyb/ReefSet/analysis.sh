#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m Hyb.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'RS_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --aut_wght_pth './result/ReefSet/AMAuT/TTDA' \
    --hub_wght_path './result/ReefSet/HuBERT/TTDA' \
    --orig_aut_wght_pth './result/ReefSet/AMAuT/train' \
    --orig_hub_wght_pth './result/ReefSet/HuBERT/train'