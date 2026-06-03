#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'AuT_US8-C_TTN_analysis.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr_momentum 0.1 \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train'