#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' --wandb