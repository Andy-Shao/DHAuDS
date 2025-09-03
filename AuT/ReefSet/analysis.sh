#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

## Analysis
python -m AuT.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 33 --output_file_name 'ReefSet_analysis.csv' \
    --orig_wght_pth './result/ReefSet/AMAuT/train' \
    --adpt_wght_path './result/ReefSet/AMAuT/TTDA'