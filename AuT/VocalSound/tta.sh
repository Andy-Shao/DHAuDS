#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
#     --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L1' \
#     --nucnm_rate 1.0 --ent_rate '1e-2' --gent_rate 0.25 --gent_q 1.1 \
#     --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
#     --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --wandb

python -m AuT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --aut_lr_decay 0.55 --num_workers 16 \
    --lr_gamma 30 --lr_threshold 35 --corruption_type 'WHN' --corruption_level 'L2' \
    --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.25 --gent_q 0.9 \
    --aut_wght_pth './result/VocalSound/AMAuT/train/aut-VS.pt' \
    --clsf_wght_pth './result/VocalSound/AMAuT/train/clsf-VS.pt' --wandb