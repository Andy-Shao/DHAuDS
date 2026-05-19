#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.6 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train'

python -m ablation_study.HuBERT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --model_level 'base' --output_file_name 'fix-corruption_analysis.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 02
python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.6 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' --seed 123456

python -m ablation_study.HuBERT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --model_level 'base' --output_file_name 'fix-corruption_analysis02.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 03 
python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.6 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' --seed 654321

python -m ablation_study.HuBERT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --model_level 'base' --output_file_name 'fix-corruption_analysis03.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 04
python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.6 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' --seed 891011

python -m ablation_study.HuBERT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --model_level 'base' --output_file_name 'fix-corruption_analysis04.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'

# Experiment 05
python -m ablation_study.HuBERT.hub_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 20 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.6 --mse_rate 0.8 \
    --lr_momentum 0.75 --hub_lr_decay 0.55 \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' --seed 111089

python -m ablation_study.HuBERT.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' --output_path './result' \
    --batch_size 32 --model_level 'base' --output_file_name 'fix-corruption_analysis05.csv' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path $BASE_PATH'/tmp'