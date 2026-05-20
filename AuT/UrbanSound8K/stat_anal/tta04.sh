#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=891011
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='US8_AuT-04.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'ENSC-L1\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --noise_path $BASE_PATH'/data' \
    --corruption_type 'ENSC' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --noise_path $BASE_PATH'/data' \
    --corruption_type 'ENSC' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'WHN' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 1.0 --lr_momentum 0.90 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 1.0 --gent_q 1.6 --mse_rate 1.0 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'WHN-L2\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'WHN' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'TST' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'TST' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'PSH' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 15 --lr '1e-4' --aut_lr_decay 0.55 --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m AuT.UrbanSound8K.ttda --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --corruption_type 'PSH' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --batch_size 70 \
    --max_epoch 30 --lr '1e-4' --lr_momentum 0.70 \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --mse_rate 1.0 \
    --aut_wght_pth './result/UrbanSound8K/AMAuT/train/aut-US8.pt' \
    --clsf_wght_pth './result/UrbanSound8K/AMAuT/train/clsf-US8.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m AuT.UrbanSound8K.analysis --dataset 'UrbanSound8K' --dataset_root_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --output_file_name $ANAL_FILE \
    --orig_wght_pth './result/UrbanSound8K/AMAuT/train' \
    --adpt_wght_path './result/UrbanSound8K/AMAuT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE