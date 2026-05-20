#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
# python -m ablation_study.AuT.aut_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 30 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
#     --lr_momentum 0.75 --aut_lr_decay 0.55 \
#     --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train'

# python -m ablation_study.AuT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
#     --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
#     --batch_size 32 --output_file_name 'fix-corruption_analysis.csv' \
#     --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
#     --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 02
python -m ablation_study.AuT.aut_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
    --lr_momentum 0.75 --aut_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 123456

python -m ablation_study.AuT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --output_file_name 'fix-corruption_analysis02.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 03
python -m ablation_study.AuT.aut_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
    --lr_momentum 0.75 --aut_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 654321

python -m ablation_study.AuT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --output_file_name 'fix-corruption_analysis03.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 04
python -m ablation_study.AuT.aut_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
    --lr_momentum 0.75 --aut_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 891011

python -m ablation_study.AuT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --output_file_name 'fix-corruption_analysis04.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 05
python -m ablation_study.AuT.aut_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
    --lr_momentum 0.75 --aut_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 111089

python -m ablation_study.AuT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --output_file_name 'fix-corruption_analysis05.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'