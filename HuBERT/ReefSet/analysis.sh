#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

## Generate Corruption sets
python -m HuBERT.ReefSet.corrupt_set --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --batch_size 33 --output_path $BASE_PATH'/data/ReefSet-C' --seed 123456 --num_workers 8 \
    --enq_path $BASE_PATH'/data/QUT-NOISE' \
    --end_path $BASE_PATH'/data/DEMAND_16k' \
    --ensc_path $BASE_PATH'/data'

## Analysis
python -m HuBERT.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 33 --output_file_name 'ReefSet_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/ReefSet/HuBERT/train' \
    --adpt_wght_path './result/ReefSet/HuBERT/TTDA'