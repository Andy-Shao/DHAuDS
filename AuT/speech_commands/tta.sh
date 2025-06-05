#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --max_epoch 30 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.25 \
    --background_path $BASE_PATH'/data/speech_commands_v0.02/speech_commands_v0.02' \
    --vocalsound_path $BASE_PATH'/data/vocalsound_16k' \
    --cochlscene_path $BASE_PATH'/data/CochlScene' \
    --corruption_level 3.0 --corruption_type 'doing_the_dishes' --fbnm_rate 1.0 --lr_gamma 30 \
    --origin_auT_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV2/AuT/train/AuT-SC2-cls0.pt' --wandb