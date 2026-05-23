#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=123456
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='ReefSet_analysis-02.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.55 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 1 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 10 --lr_threshold 1 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENSC' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'PSH' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'TST' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'ENQ' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 --lr_momentum 0.70 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 2.0 --corruption_type 'END1' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.70 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END1' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '5e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L2' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m HuBERT.ReefSet.ttda --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data/DEMAND_16k' \
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-5' --hub_lr_decay 0.35 --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 --lr_momentum 0.75 --mse_rate 0.1 \
    --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.1 --corruption_type 'END2' --corruption_level 'L1' \
    --hub_wght_pth './result/ReefSet/HuBERT/train/hubert-base-RS.pt' \
    --clsf_wght_pth './result/ReefSet/HuBERT/train/clsModel-base-RS.pt' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m HuBERT.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 33 --output_file_name $ANAL_FILE --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth './result/ReefSet/HuBERT/train' \
    --adpt_wght_path './result/ReefSet/HuBERT/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE