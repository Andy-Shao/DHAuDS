#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m HuBERT.CochlScene.ttda --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 15 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --hub_lr_decay 0.45 --num_workers 16 \
    --nucnm_rate 0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 1.0 --gent_rate 0 --gent_q 1.1 --corruption_type 'WHN' --corruption_level 'L2' \
    --hub_wght_pth './result/CochlScene/HuBERT/train/hubert-base-CS.pt' \
    --clsf_wght_pth './result/CochlScene/HuBERT/train/clsModel-base-CS.pt'