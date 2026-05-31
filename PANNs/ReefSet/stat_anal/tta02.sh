#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=123456
export LOG_FILE=$BASE_PATH'/DHAuDS.log'
export ANAL_FILE='PAN_RS-C_analysis-02.csv'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on RS-C, seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'WHN-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'WHN' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.2 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'WHN-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'WHN' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.2 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'ENQ-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'ENQ' --lr 1e-4 --max_epoch 25 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'ENQ-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'ENQ' --lr 1e-4 --max_epoch 25 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'END1-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'END1' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.75 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'END1-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'END1' --lr 1e-5 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.70 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'END2-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'END2' --lr 1e-4 --max_epoch 25 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'END2-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'END2' --lr 1e-4 --max_epoch 25 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'ENSC-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'ENSC' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.75 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'ENSC-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'ENSC' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.75 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --seed $SEED_VAL

printf 'PSH-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'PSH' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 --lr_momentum 0.75 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'PSH-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'PSH' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 --lr_momentum 0.75 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'TST-L2\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'TST' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'TST-L1\n' >> $LOG_FILE
python -m PANNs.ReefSet.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L1' --corruption_type 'TST' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --seed $SEED_VAL

printf 'TTA processing is finished\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE
printf 'Output analysis file to '$ANAL_FILE'\n' >> $LOG_FILE

python -m PANNs.ReefSet.analysis --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet-C' \
    --output_file_name $ANAL_FILE --batch_size 32 \
    --orig_wght_pth './result/ReefSet/PANNs/train' \
    --adpt_wght_pth './result/ReefSet/PANNs/TTDA'