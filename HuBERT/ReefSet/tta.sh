#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate '1e-3' --gent_rate '1e-3' --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate '1e-3' --gent_rate '1e-3' --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
#     --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.0 --corruption_type 'END1' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.25 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L2' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb

# python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
#     --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L1' \
#     --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
#     --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --wandb
