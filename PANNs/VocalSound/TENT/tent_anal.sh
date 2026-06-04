#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
#     --output_file_name 'PAN_VS-C_TENT_analysis.csv' \
#     --orig_wght_pth './result/VocalSound/PANNs/train'

python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_VS-C_TENT_analysis-02.csv' \
    --orig_wght_pth './result/VocalSound/PANNs/train' --seed 123456

python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_VS-C_TENT_analysis-03.csv' \
    --orig_wght_pth './result/VocalSound/PANNs/train' --seed 654321

python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_VS-C_TENT_analysis-04.csv' \
    --orig_wght_pth './result/VocalSound/PANNs/train' --seed 891011

python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_VS-C_TENT_analysis-05.csv' \
    --orig_wght_pth './result/VocalSound/PANNs/train' --seed 111098