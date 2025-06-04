#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.prepare_data --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --background_path $BASE_PATH'/data/speech_commands_v0.02/speech_commands_v0.02' \
    --vocalsound_path $BASE_PATH'/data/vocalsound_16k' \
    --cochlscene_path $BASE_PATH'/data/CochlScene' \
    --output_path $BASE_PATH'/tmp' --corruption_level 3.0 --corruption_type 'doing_the_dishes'