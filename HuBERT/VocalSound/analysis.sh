#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

## Generate Corruption sets
python -m HuBERT.VocalSound.corrupt_set --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 33 --output_path $BASE_PATH'/data/VocalSound-C' --seed 123456 --num_workers 8 \
    --enq_path $BASE_PATH'/data/QUT-NOISE' \
    --end_path $BASE_PATH'/data/DEMAND_16k' \
    --ensc_path $BASE_PATH'/data'

## Analysis
python -m HuBERT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --output_file_name 'VocalSound_analysis.csv' --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/VocalSound/HuBERT/train' \
    --adpt_wght_path './result/VocalSound/HuBERT/TTDA'