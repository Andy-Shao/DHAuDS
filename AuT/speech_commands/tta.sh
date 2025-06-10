#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --max_epoch 200 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.55 \
    --corruption_level 3.0 --corruption_type 'doing_the_dishes' --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 \
    --origin_auT_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-cls0.pt' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' --wandb