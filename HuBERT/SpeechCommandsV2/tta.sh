#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data' \
#     --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 25 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L2' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 25 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L1' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
#     --max_epoch 20 --lr_cardinality 40 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 1 \
#     --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L2' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb

# python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data' \
#     --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
#     --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
#     --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --wandb