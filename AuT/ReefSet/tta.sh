#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --corruption_type 'END1' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr '1e-4' \
    --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
    --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt'