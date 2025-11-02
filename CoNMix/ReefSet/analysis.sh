#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 33 --output_file_name 'ReefSet_analysis.csv' --normalized \
    --orig_wght_pth './result/ReefSet/CoNMix/train' \
    --adpt_wght_path './result/ReefSet/CoNMix/STDA'