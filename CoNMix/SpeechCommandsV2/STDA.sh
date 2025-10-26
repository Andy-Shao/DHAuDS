#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.SpeechCommandsV2.STDA --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --test_batch_size 96 --max_epoch 50 --interval 50 --lr '1e-4' \
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-doing_the_dishes.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-doing_the_dishes.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-doing_the_dishes.pt' --normalized \
    --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --plr 1 \
    --alpha 0.9 --initc_num 1 --cls_mode 'logsoft_nll' --lr_gamma 30 \
    --corruption_type 'WHN' --corruption_level 'L1' --cache_path $BASE_PATH'/tmp'