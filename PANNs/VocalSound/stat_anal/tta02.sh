#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=123456
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='PAN_VS-C_analysis-02.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on PAN VS-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'WHN' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'WHN' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENQ' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'ENDQ-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENQ' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END1' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
    --lr 5e-6 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --pan_lr_decay 1.0 --lr_momentum 0.75 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END1' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 5e-6 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --pan_lr_decay 1.0 --lr_momentum 0.75 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END2' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
    --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --seed $SEED_VAL

prinft 'END2-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END2' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENSC' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

prinft 'ENSC-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENSC' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

prinft 'PSH-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'PSH' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'PSH' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'TST' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'TST' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m PANNs.VocalSound.analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/VocalSound-C' \
    --output_file_name $ANAL_FILE --batch_size 32 \
    --orig_wght_pth './result/VocalSound/PANNs/train' \
    --adpt_wght_pth './result/VocalSound/PANNs/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE