#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 32 --output_file_name 'VocalSound_analysis.csv' \
    --orig_wght_pth './result/VocalSound/AMAuT/train'