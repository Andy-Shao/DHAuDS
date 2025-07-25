#!bin/bash
export BASE_PATH='/root'

python -m HuBERT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --cache_path $BASE_PATH'/tmp' --repeat_time 3 \
    --corruption_type 'WHN' --corruption_level 'L2' --batch_size 32 \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' \
    --adpt_hub_wght_pth './result/VocalSound/HuBERT/TTDA/hubert-base-VS-WHN-L2.pt' \
    --adpt_clsf_wght_pth './result/VocalSound/HuBERT/TTDA/clsModel-base-VS-WHN-L2.pt'