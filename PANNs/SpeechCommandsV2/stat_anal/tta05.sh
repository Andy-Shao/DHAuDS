#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=111098
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='PAN_SC2-C_analysis-05.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on SC2-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'WHN' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.1 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'WHN' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 3.1 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'ENQ' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'ENQ' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.75 --pan_lr_decay 1.0 --gent_q 5.1 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'END1' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.70 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'END1' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.75 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'END2' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.75 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'END2' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.70 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'ENSC' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'ENSC' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 3.1 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'PSH' --corruption_level 'L2' --batch_size 70 --max_epoch 60 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 --lr_cardinality 80 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'PSH' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 --lr_cardinality 80 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'TST' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m PANNs.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'TST' --corruption_level 'L1' --batch_size 70 --max_epoch 25 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m PANNs.SpeechCommandsV2.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --output_file_name $ANAL_FILE \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/PANNs/TTDA'

printf 'ALL processing is finished\n' >> $LOG_FILE