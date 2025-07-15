#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AST.speech_commands.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --noise_path $BASE_PATH'/data/QUT-NOISE' --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 200 --lr_cardinality 50 --batch_size 70 --lr '5.5e-5' --num_workers 16 \
#     --nucnm_rate 0.1 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 1.0 --gent_rate 1.0 --gent_q 1.1 --corruption_type 'ENQP' --corruption_level 'L2' --wandb

# python -m AST.speech_commands.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --noise_path $BASE_PATH'/data/QUT-NOISE' --cache_path $BASE_PATH'/tmp' \
#     --max_epoch 200 --lr_cardinality 50 --batch_size 70 --lr '5.5e-5' --num_workers 16 \
#     --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
#     --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'ENQP' --corruption_level 'L1' --wandb

python -m AST.speech_commands.ttda --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 120 --lr_cardinality 50 --batch_size 70 --lr '5.5e-5' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHNP' --corruption_level 'L2' --wandb