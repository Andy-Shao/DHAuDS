#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.VocalSound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --max_epoch 35 --batch_size 32 --lr '1e-3' --validation_mode 'validation' \
    --lr_cardinality 80 --background_path $BASE_PATH'/data/speech_commands_v0.01' --wandb