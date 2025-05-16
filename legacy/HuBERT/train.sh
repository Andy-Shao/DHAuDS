export BASE_PATH=${BASE_PATH:-'/root'}

python -m train --dataset_root_path $BASE_PATH'/data/speech_commands_v0.02' \
    --batch_size 32