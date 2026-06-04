#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.VocalSound.TENT.tent_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --adpt_batch_size 70 --lr 1e-3 \
    --output_file_name 'PAN_VS-C_TENT_analysis.csv' \
    --orig_wght_pth './result/VocalSound/PANNs/train'