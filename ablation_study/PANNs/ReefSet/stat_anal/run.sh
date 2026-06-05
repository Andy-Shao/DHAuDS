#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
# python -m ablation_study.PANNs.ReefSet.pan_fix_analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 10 --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.0 --mse_rate 0.1 \
#     --lr_momentum 0.75 --pan_lr_decay 0.55 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --wandb

python -m ablation_study.PANNs.ReefSet.stat_anal.analysis --dataset 'ReefSet' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
    --output_file_name 'fix-corruption_analysis.csv' --batch_size 32 \
    --orig_wght_pth './result/ReefSet/PANNs/train' \
    --adpt_wght_pth $BASE_PATH'/tmp'