#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=111098
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='UrbanSound8K_analysis-05.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.05 --lr_momentum 0.75 \
    --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
    --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.25 --gent_rate 0.25 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L1' \
    --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 40 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
    --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 40 --lr_threshold 35 --lr_momentum 0.70 --mse_rate 0.1 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L1' \
    --hub_wght_pth './result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset UrbanSound8K --dataset_root_path /root/data/Ada-UrbanSound8K-C \
    --cache_path /root/tmp \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr 5e-5 --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.70 --mse_rate 0.1 --ent_rate 0.01 \
    --gent_rate 0.01 --gent_q 1.1 --corruption_type PSH --corruption_level L2 \
    --hub_wght_pth ./result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt \
    --clsf_wght_pth ./result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset UrbanSound8K --dataset_root_path /root/data/Ada-UrbanSound8K-C \
    --cache_path /root/tmp \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr 5e-5 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type PSH --corruption_level L1 \
    --hub_wght_pth ./result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt \
    --clsf_wght_pth ./result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset UrbanSound8K --dataset_root_path /root/data/Ada-UrbanSound8K-C \
    --cache_path /root/tmp \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr 1e-4 --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type TST --corruption_level L2 \
    --hub_wght_pth ./result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt \
    --clsf_wght_pth ./result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m HuBERT.UrbanSound8K.ttda --dataset UrbanSound8K --dataset_root_path /root/data/Ada-UrbanSound8K-C \
    --cache_path /root/tmp \
    --max_epoch 10 --lr_cardinality 50 --batch_size 33 --lr 1e-4 --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.01 --gent_rate 0.01 --gent_q 1.1 --corruption_type TST --corruption_level L1 \
    --hub_wght_pth ./result/UrbanSound8K/HuBERT/train/hubert-base-US8.pt \
    --clsf_wght_pth ./result/UrbanSound8K/HuBERT/train/clsModel-base-US8.pt --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m HuBERT.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 33 --output_file_name $ANAL_FILE --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/UrbanSound8K/HuBERT/train' \
    --adpt_wght_path './result/UrbanSound8K/HuBERT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE