#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m PANNs.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name 'PAN_VS-C_analysis.csv' --batch_size 32 \
    --orig_wght_pth './result/VocalSound/PANNs/train' \
    --adpt_wght_pth './result/VocalSound/PANNs/TTDA'