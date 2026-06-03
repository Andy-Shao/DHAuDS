#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m PANNs.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --batch_size 32 --adpt_batch_size 70 --lr_momentum 0.1 \
#     --output_file_name 'PAN_US8-C_TTN_analysis.csv' \
#     --orig_wght_pth './result/UrbanSound8K/PANNs/train'

python -m PANNs.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --adpt_batch_size 70 --lr_momentum 0.1 \
    --output_file_name 'PAN_US8-C_TTN_analysis-02.csv' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' --seed 123456

python -m PANNs.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --adpt_batch_size 70 --lr_momentum 0.1 \
    --output_file_name 'PAN_US8-C_TTN_analysis-03.csv' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' --seed 654321

python -m PANNs.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --adpt_batch_size 70 --lr_momentum 0.1 \
    --output_file_name 'PAN_US8-C_TTN_analysis-04.csv' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' --seed 891011

python -m PANNs.UrbanSound8K.TTN.ttn_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --adpt_batch_size 70 --lr_momentum 0.1 \
    --output_file_name 'PAN_US8-C_TTN_analysis-05.csv' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' --seed 111098