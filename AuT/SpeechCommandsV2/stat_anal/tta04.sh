#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=891011
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='SpeechCommandsV2_analysis-04.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' \
   --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' \
   --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.5 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENSC' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 --mse_rate 0.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENSC' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'PSH' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.05 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
    --lr_gamma 30 --lr_threshold 35 --corruption_type 'PSH' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.6 --mse_rate 0.05 \
    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'TST' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.6 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
  --cache_path $BASE_PATH'/tmp' \
  --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
  --lr_gamma 30 --lr_threshold 35 --corruption_type 'TST' --corruption_level 'L2' \
  --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.6 --mse_rate 1.0 \
  --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
  --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENQ' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --lr_momentum 0.75 --mse_rate 0.05 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENQ' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.1 --gent_q 2.5 --lr_momentum 0.75 --mse_rate 1.0  \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'END1' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --lr_momentum 0.75 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'END1' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --lr_momentum 0.75 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 60 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
   --lr_gamma 30 --lr_threshold 35 --corruption_type 'END2' --corruption_level 'L1' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --lr_momentum 0.75 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
   --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
   --max_epoch 40 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 \
   --lr_gamma 20 --lr_threshold 35 --corruption_type 'END2' --corruption_level 'L2' \
   --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 1.0 --gent_q 2.0 --mse_rate 1.0 \
   --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
   --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m AuT.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --output_file_name $ANAL_FILE \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path './result/SpeechCommandsV2/AMAuT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE