#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m PANNs.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --output_file_name 'PAN_RS-C_TENT_analysis.csv' --batch_size 32 \
#     --lr 1e-5 --adpt_batch_size 70 \
#     --orig_wght_pth './result/ReefSet/PANNs/train'

python -m PANNs.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_TENT_analysis-02.csv' --batch_size 32 \
    --lr 1e-5 --adpt_batch_size 70 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed 123456

python -m PANNs.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_TENT_analysis-03.csv' --batch_size 32 \
    --lr 1e-5 --adpt_batch_size 70 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed 654321

python -m PANNs.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_TENT_analysis-04.csv' --batch_size 32 \
    --lr 1e-5 --adpt_batch_size 70 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed 891011

python -m PANNs.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name 'PAN_RS-C_TENT_analysis-05.csv' --batch_size 32 \
    --lr 1e-5 --adpt_batch_size 70 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed 111098