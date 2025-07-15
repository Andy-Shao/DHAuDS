#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AST.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --noise_path $BASE_PATH'/data/QUT-NOISE' --cache_path $BASE_PATH'/tmp' \
#     --corruption_type 'ENQP' --corruption_level 'L2' --num_workers 16 --batch_size 70 \
#     --adpt_wght_pth './result/SpeechCommandsV2/AST/TTDA/AST-SC2-ENQP-L2.pt'

# python -m AST.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
#     --cache_path $BASE_PATH'/tmp' \
#     --corruption_type 'WHNP' --corruption_level 'L2' --num_workers 16 --batch_size 70 \
#     --adpt_wght_pth './result/SpeechCommandsV2/AST/TTDA/AST-SC2-WHNP-L2.pt'

python -m AST.speech_commands.analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --noise_path $BASE_PATH'/data/DEMAND_16k' --cache_path $BASE_PATH'/tmp' \
    --corruption_type 'ENDP1' --corruption_level 'L2' --num_workers 16 --batch_size 70 \
    --adpt_wght_pth './result/SpeechCommandsV2/AST/TTDA/AST-SC2-ENDP1-L2.pt'