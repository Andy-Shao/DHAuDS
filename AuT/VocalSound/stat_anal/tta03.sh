#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=654321
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='VocalSound_analysis-03.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'WHN' --corruption_level 'L1' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'WHN-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'WHN' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 0.25 --gent_rate 0.0 --gent_q 0.9 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'ENSC' --corruption_level 'L1' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'ENSC' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 1.0 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp'\
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'PSH' --corruption_level 'L1' \
    --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 35 --corruption_type 'PSH' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 35 --corruption_type 'TST' --corruption_level 'L1' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' \
   --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
   --lr_gamma 10 --lr_threshold 15 --corruption_type 'TST' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
   --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
   --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 35 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'END1' --corruption_level 'L1' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
   --lr_gamma 10 --lr_threshold 15 --corruption_type 'END1' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
   --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
   --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
   --lr_gamma 10 --lr_threshold 15 --corruption_type 'END2' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
   --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
   --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
   --lr_gamma 10 --lr_threshold 15 --corruption_type 'END2' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
   --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
   --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
   --max_epoch 45 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
   --lr_gamma 10 --lr_threshold 15 --corruption_type 'ENQ' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
   --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
   --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 1.0 --num_workers 16 \
    --lr_gamma 10 --lr_threshold 15 --corruption_type 'ENQ' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.1 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m AuT.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --output_file_name $ANAL_FILE \
    --orig_wght_pth './result/VocalSound/AMAuT/train' \
    --adpt_wght_path './result/VocalSound/AMAuT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE