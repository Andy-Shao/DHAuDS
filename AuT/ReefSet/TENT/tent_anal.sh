#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.ReefSet.TENT.tent_anal --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C'\
    --eval_set_path $BASE_PATH'/data/ReefSet-C'\
    --batch_size 32 --output_file_name 'AuT_RS-C_TENT_analysis.csv' \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/ReefSet/AMAuT/train'