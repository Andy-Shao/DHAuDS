#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 0.55 --num_workers 16 \
#     --nucnm_rate 0.1 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
#     --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
#     --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --wandb

# python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 0.55 --num_workers 16 \
#     --nucnm_rate 0.1 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L2' \
#     --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
#     --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --wandb

python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 0.55 --num_workers 16 \
    --nucnm_rate 0.1 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --wandb