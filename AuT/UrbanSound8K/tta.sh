#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
#     --noise_path $BASE_PATH'/data' \
#     --corruption_type 'ENSC' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 30 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
#     --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --wandb

# python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
#     --noise_path $BASE_PATH'/data' \
#     --corruption_type 'ENSC' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 30 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
#     --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --wandb

python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --corruption_type 'WHN' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --wandb