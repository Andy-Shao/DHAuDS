#!bin/bash
export BASE_PATH=${BASE_PATH-:'/root'}
report_list="\
./result/SpeechCommandsV2/HuBERT/Analysis/SpeechCommandsV2_analysis.csv, \
./result/SpeechCommandsV2/HuBERT/Analysis/SpeechCommandsV2_analysis02.csv, \
./result/SpeechCommandsV2/HuBERT/Analysis/SpeechCommandsV2_analysis03.csv"

python -m HuBERT.SpeechCommandsV2.stat_anal.stat_anal --output_file_name 'SC2-C_statistic_analysis.csv' \
    --report_list "${report_list}"