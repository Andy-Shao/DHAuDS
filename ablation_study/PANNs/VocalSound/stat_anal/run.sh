#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
python -m ablation_study.PANNs.VocalSound.pan_fix_analysis --dataset 'VocalSound' \
    --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.0 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb