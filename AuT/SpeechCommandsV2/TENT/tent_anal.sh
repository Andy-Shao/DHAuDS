#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --output_file_name 'AuT_SC2-C_TENT_analysis.csv' --batch_size 32 \
#     --adpt_batch_size 70 --lr 1e-5 \
#     --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train'

python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'AuT_SC2-C_TENT_analysis-02.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 123456

python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'AuT_SC2-C_TENT_analysis-03.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 654321

python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'AuT_SC2-C_TENT_analysis-04.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 891011

python -m AuT.SpeechCommandsV2.TENT.tent_anal --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file_name 'AuT_SC2-C_TENT_analysis-05.csv' --batch_size 32 \
    --adpt_batch_size 70 --lr 1e-5 \
    --orig_wght_pth './result/SpeechCommandsV2/AMAuT/train' --seed 111098