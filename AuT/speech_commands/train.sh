#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.speech_commands.train --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --background_path $BASE_PATH'/data/speech_commands_v0.02/speech_commands_v0.02' --file_name_suffix '0' \
#     --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --num_workers 16 --wandb

python -m AuT.speech_commands.train --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' --file_name_suffix '0' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --num_workers 16 --wandb

python -m AuT.speech_commands.train --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' --file_name_suffix '1' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --num_workers 16

python -m AuT.speech_commands.train --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' --file_name_suffix '2' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --num_workers 16