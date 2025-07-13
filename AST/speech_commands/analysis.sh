#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AST.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --noise_path $BASE_PATH'/data/QUT-NOISE' --cache_path $BASE_PATH'/tmp' \
#     --corruption_type 'ENQ' --corruption_level 'L2' --num_workers 16 --batch_size 70 \
#     --adpt_wght_pth './result/SpeechCommandsV2/AST/TTDA/AST-SC2-ENQ-L2.pt'

python -m AST.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --cache_path $BASE_PATH'/tmp' \
    --corruption_type 'WHN' --corruption_level 'L2' --num_workers 16 --batch_size 70 \
    --adpt_wght_pth './result/SpeechCommandsV2/AST/TTDA/AST-SC2-WHN-L2.pt'