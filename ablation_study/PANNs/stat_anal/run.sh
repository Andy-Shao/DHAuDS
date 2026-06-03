#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Experiment 01
# python -m ablation_study.PANNs.pan_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
#     --max_epoch 23 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.0 --mse_rate 1.0 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

# python -m ablation_study.PANNs.stat_anal.analysis --dataset 'SpeechCommandsV2' \
#     --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
#     --output_file_name 'fix-corruption_analysis.csv' --batch_size 32 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
#     --adpt_wght_pth $BASE_PATH'/tmp'

# Experiment 02
python -m ablation_study.PANNs.pan_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 23 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.0 --mse_rate 1.0 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed 123456

python -m ablation_study.PANNs.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
    --output_file_name 'fix-corruption_analysis02.csv' --batch_size 32 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth $BASE_PATH'/tmp'

# Experiment 03
python -m ablation_study.PANNs.pan_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 23 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.0 --mse_rate 1.0 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed 654321

python -m ablation_study.PANNs.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
    --output_file_name 'fix-corruption_analysis03.csv' --batch_size 32 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth $BASE_PATH'/tmp'

# Experiment 04
python -m ablation_study.PANNs.pan_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 23 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.0 --mse_rate 1.0 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed 891011

python -m ablation_study.PANNs.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
    --output_file_name 'fix-corruption_analysis04.csv' --batch_size 32 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth $BASE_PATH'/tmp'

# Experiment 05
python -m ablation_study.PANNs.pan_fix_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/DEMAND_16k' --output_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 23 --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.0 --mse_rate 1.0 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed 111098

python -m ablation_study.PANNs.stat_anal.analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/tmp/fix_clip_set' \
    --output_file_name 'fix-corruption_analysis05.csv' --batch_size 32 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth $BASE_PATH'/tmp'