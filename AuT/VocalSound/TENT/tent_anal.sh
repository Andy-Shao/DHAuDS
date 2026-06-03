#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --output_file_name 'AuT_VS-C_TENT_analsysis.csv' --batch_size 32 \
#     --adpt_batch_size 70 --lr 1e-3 \
#     --orig_wght_pth './result/VocalSound/AMAuT/train'

python -m AuT.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'AuT_VS-C_TENT_analsysis-02.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/VocalSound/AMAuT/train' --seed 123456

python -m AuT.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'AuT_VS-C_TENT_analsysis-03.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/VocalSound/AMAuT/train' --seed 654321

python -m AuT.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'AuT_VS-C_TENT_analsysis-04.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/VocalSound/AMAuT/train' --seed 891011

python -m AuT.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'AuT_VS-C_TENT_analsysis-05.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-3 \
    --orig_wght_pth './result/VocalSound/AMAuT/train' --seed 111098