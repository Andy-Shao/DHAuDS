#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=891011
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='VocalSound_analysis-04.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 0.1 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'END-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
   --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.75 \
   --ent_rate 0.1 --gent_rate 0.1 --gent_q 3.1 --corruption_type 'END1' --corruption_level 'L2' \
   --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
   --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
   --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.75 \
   --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L1' \
   --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
   --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr 1e-4 --hub_lr_decay 1.0 --num_workers 16 \
   --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
   --ent_rate 0.1 --gent_rate 0.1 --gent_q 3.1 --corruption_type 'END2' --corruption_level 'L2' \
   --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
   --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.70 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.1 --gent_q 3.1 --corruption_type 'ENSC' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 25 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 32 --lr '5e-5' --hub_lr_decay 0.55 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 1.0 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 15 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L1' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m HuBERT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --output_file_name $ANAL_FILE --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/VocalSound/HuBERT/train' \
    --adpt_wght_path './result/VocalSound/HuBERT/TTDA'
