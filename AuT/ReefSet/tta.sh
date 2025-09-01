#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --corruption_type 'END1' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --corruption_type 'END1' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '5e-5' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --corruption_type 'END2' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --corruption_type 'END2' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '5e-5' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data' \
#     --corruption_type 'ENSC' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '5e-5' --aut_lr_decay 0.45 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data' \
#     --corruption_type 'ENSC' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '5e-5' --aut_lr_decay 0.45 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.1 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --corruption_type 'WHN' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

# python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --corruption_type 'WHN' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
#     --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 \
#     --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb

python -m AuT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --corruption_type 'ENQ' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '5e-5' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 \
    --aut_wght_pth './result/ReefSet/AMAuT/train/aut-RS.pt' \
    --clsf_wght_pth './result/ReefSet/AMAuT/train/clsf-RS.pt' --wandb