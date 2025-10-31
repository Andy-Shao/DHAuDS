#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.ReefSet.STDA --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --batch_size 33 --max_epoch 50 --interval 50 --early_stop 15 --lr '1e-4' \
#     --modelF_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelF.pt' \
#     --modelB_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelB.pt' \
#     --modelC_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelC.pt' \
#     --normalized --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --plr 1 \
#     --alpha 0.9 --initc_num 1 --cls_mode 'logsoft_nll' --lr_gamma 30 \
#     --corruption_type 'WHN' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --wandb

# python -m CoNMix.ReefSet.STDA --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --batch_size 33 --max_epoch 50 --interval 50 --early_stop 15 --lr '1e-4' \
#     --modelF_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelF.pt' \
#     --modelB_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelB.pt' \
#     --modelC_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelC.pt' \
#     --normalized --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --plr 1 \
#     --alpha 0.9 --initc_num 1 --cls_mode 'logsoft_nll' --lr_gamma 30 \
#     --corruption_type 'WHN' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --wandb

# python -m CoNMix.ReefSet.STDA --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
#     --noise_path $BASE_PATH'/data/QUT-NOISE' \
#     --batch_size 33 --max_epoch 50 --interval 50 --early_stop 15 --lr '1e-4' --lr_momentum 0.75 \
#     --modelF_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelF.pt' \
#     --modelB_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelB.pt' \
#     --modelC_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelC.pt' \
#     --normalized --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --plr 1 \
#     --alpha 0.9 --initc_num 1 --cls_mode 'logsoft_nll' --lr_gamma 30 \
#     --corruption_type 'ENQ' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp' --wandb

python -m CoNMix.ReefSet.STDA --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --noise_path $BASE_PATH'/data/QUT-NOISE' \
    --batch_size 33 --max_epoch 50 --interval 50 --early_stop 15 --lr '1e-4' --lr_momentum 0.75 \
    --modelF_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelF.pt' \
    --modelB_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelB.pt' \
    --modelC_weight_path './result/ReefSet/CoNMix/train/CoN-RS-modelC.pt' \
    --normalized --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --plr 1 \
    --alpha 0.9 --initc_num 1 --cls_mode 'logsoft_nll' --lr_gamma 30 \
    --corruption_type 'ENQ' --corruption_level 'L2' --cache_path $BASE_PATH'/tmp' --wandb