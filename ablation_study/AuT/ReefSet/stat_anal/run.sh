#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
python -m ablation_study.AuT.ReefSet.aut_fix_analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 10 --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.0 --mse_rate 0.5 \
    --lr_momentum 0.70 \
    --orig_wght_pth './result/ReefSet/AMAuT/train' --wandb