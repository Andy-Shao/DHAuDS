#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Generate data
python -m HuBERT.UrbanSound8K.corrupt_set --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --output_path $BASE_PATH'/data/UrbanSound8K-C' --batch_size 33 --seed 123456 --ensc_path $BASE_PATH'/data'