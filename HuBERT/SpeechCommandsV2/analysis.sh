#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --cache_path $BASE_PATH'/tmp' --repeat_time 3 --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --corruption_type 'END1' --corruption_level 'L1' --batch_size 32 \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' \
    --adpt_hub_wght_pth './result/SpeechCommandsV2/HuBERT/TTDA/hubert-base-SC2-END1-L1.pt' \
    --adpt_clsf_wght_pth './result/SpeechCommandsV2/HuBERT/TTDA/clsModel-base-SC2-END1-L1.pt'