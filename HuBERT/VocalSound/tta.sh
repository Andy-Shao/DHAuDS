#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.VocalSound.ttda --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 5 --lr_cardinality 50 --batch_size 32 --lr '5.5e-5' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHNP' --corruption_level 'L2' \
    --hub_wght_pth './result/VocalSound/HuBERT/train/hubert-base-VS.pt' \
    --clsf_wght_pth './result/VocalSound/HuBERT/train/clsModel-base-VS.pt' --wandb