#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --output_file_name 'AuT_US8-C_TENT_analysis.csv' --batch_size 32 \
#     --adpt_batch_size 70 --lr 1e-5 \
#     --orig_wght_pth './result/UrbanSound8K/AMAuT/train'

python -m AuT.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'AuT_US8-C_TENT_analysis-02.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' --seed 123456

python -m AuT.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'AuT_US8-C_TENT_analysis-03.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' --seed 654321

python -m AuT.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'AuT_US8-C_TENT_analysis-04.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' --seed 891011

python -m AuT.UrbanSound8K.TENT.tent_anal --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file_name 'AuT_US8-C_TENT_analysis-05.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' --seed 111098