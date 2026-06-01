#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
#     --eval_set_path $BASE_PATH'/data/ReefSet-C'\
#     --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis.csv' \
#     --adpt_batch_size 70 --lr 1e-3 \
#     --orig_wght_pth './result/ReefSet/AMAuT/train'

python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
    --eval_set_path $BASE_PATH'/data/ReefSet-C'\
    --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis-02.csv' \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/ReefSet/AMAuT/train' --seed 123456

python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
    --eval_set_path $BASE_PATH'/data/ReefSet-C'\
    --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis-03.csv' \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/ReefSet/AMAuT/train' --seed 654321

python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
    --eval_set_path $BASE_PATH'/data/ReefSet-C'\
    --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis-04.csv' \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/ReefSet/AMAuT/train' --seed 891011

python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
    --eval_set_path $BASE_PATH'/data/ReefSet-C'\
    --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis-05.csv' \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/ReefSet/AMAuT/train' --seed 111098