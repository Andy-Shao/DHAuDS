#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.analysis --dataset 'SpeechCommandsV1' --dataset_root_path $BASE_PATH'/data/speech_commands_v0.01' \
    --background_path $BASE_PATH'/data/speech_commands_v0.01' \
    --corruption_level 3.0 --analysis_file 'analysis.3.0.csv' --noise_types 'doing_the_dishes' --softmax \
    --batch_size 32 --origin_auT_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-auT0.pt' \
    --origin_cls_weight './result/SpeechCommandsV1/AuT/train/AuT-SC1-cls0.pt' \
    --adapted_auT_weight1 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-auT-DTD-3.0CEG.pt' \
    --adapted_cls_weight1 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-cls-DTD-3.0CEG.pt' \
    --adapted_auT_weight2 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-auT-DTD-3.0CE.pt' \
    --adapted_cls_weight2 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-cls-DTD-3.0CE.pt' \
    --adapted_auT_weight3 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-auT-DTD-3.0CG.pt' \
    --adapted_cls_weight3 './result/SpeechCommandsV1/AuT/TTA/AuT-SC1-cls-DTD-3.0CG.pt'