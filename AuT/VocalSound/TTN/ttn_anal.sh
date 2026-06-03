#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.VocalSound.TTN.ttn_anal --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'AuT_VS-C_TTN_analsysis.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr_momentum 0.1 \
    --orig_wght_pth './result/VocalSound/AMAuT/train'