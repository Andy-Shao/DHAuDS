export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.train --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --max_epoch 30
