#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Pre-training
# python -m AuT.UrbanSound8K.pre_train --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --wandb

# Training
python -m AuT.UrbanSound8K.train --dataset UrbanSound8K --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --background_path $BASE_PATH'/data/DEMAND_48k' --max_epoch 30 --batch_size 32 --lr '1e-3' \
    --aut_pt_wght_pth './result/UrbanSound8K/AMAuT/PreTrain/pre-aut-CS.pt' \
    --clsf_pt_wght_pth './result/UrbanSound8K/AMAuT/PreTrain/pre-clsf-CS.pt' --wandb