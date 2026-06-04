#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
python -m ablation_study.HuBERT.ReefSet.hub_fix_analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 10 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.1 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/ReefSet/HuBERT/train' --wandb