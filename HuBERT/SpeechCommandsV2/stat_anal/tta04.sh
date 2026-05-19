#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=891011
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='SpeechCommandsV2_analysis04.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.05 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.05 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 35 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.925 --mse_rate 0.3 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 35 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.925 --mse_rate 0.5 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 35 --lr_cardinality 40 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 1 --lr_momentum 0.9 --mse_rate 0.6 \
    --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 35 --lr_cardinality 40 --batch_size 70 --lr '1e-4' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 1 --lr_momentum 0.9 --mse_rate 0.5 \
    --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.5 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.5 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L2' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m HuBERT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp'  --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L1' \
    --hub_wght_pth './result/SpeechCommandsV2/HuBERT/train/hubert-base-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/HuBERT/train/clsModel-base-SC2.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m HuBERT.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name $ANAL_FILE --batch_size 32 --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/SpeechCommandsV2/HuBERT/train' \
    --adpt_wght_path './result/SpeechCommandsV2/HuBERT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE