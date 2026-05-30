#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'WHN' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'WHN' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'ENQ' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENQ' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'END1' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
#     --lr 5e-6 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --pan_lr_decay 1.0 --lr_momentum 0.75 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END1' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 5e-6 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --pan_lr_decay 1.0 --lr_momentum 0.75 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'END2' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
#     --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'END2' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'ENSC' --corruption_level 'L2' --batch_size 70 --max_epoch 30 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'ENSC' --corruption_level 'L1' --batch_size 70 --max_epoch 30 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'PSH' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'PSH' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

# python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_type 'TST' --corruption_level 'L2' --batch_size 70 --max_epoch 20 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb

python -m PANNs.VocalSound.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'TST' --corruption_level 'L1' --batch_size 70 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --max_mode --wandb