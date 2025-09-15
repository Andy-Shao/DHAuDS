#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

#python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#    --cache_path $BASE_PATH'/tmp' \
#    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
#    --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L2' \
#    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 \
#    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
#    --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --wandb

#python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#    --cache_path $BASE_PATH'/tmp' \
#    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
#    --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L1' \
#    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 \
#    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
#    --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --wandb


#python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
#    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
#    --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENSC' --corruption_level 'L1' \
#    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 \
#    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
#    --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --wandb

python -m AuT.SpeechCommandsV2.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --cache_path $BASE_PATH'/tmp' --noise_path $BASE_PATH'/data' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
    --lr_gamma 30 --lr_threshold 35 --corruption_type 'ENSC' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 1.0 --gent_rate 0.0 --gent_q 1.1 \
    --aut_wght_pth './result/SpeechCommandsV2/AMAuT/train/aut-SC2.pt' \
    --clsf_wght_pth './result/SpeechCommandsV2/AMAuT/train/clsf-SC2.pt' --wandb
