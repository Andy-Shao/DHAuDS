#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
#     --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
#     --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --wandb

python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 40 --lr_threshold 35 --lr_momentum 0.75 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
    --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --wandb

# python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
#     --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
#     --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
#     --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --wandb

# python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
#     --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
#     --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --wandb