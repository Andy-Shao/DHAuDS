#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/SpeechCommandsV2/AMAuT/Analysis/SpeechCommandsV2_analysis.csv, \
./result/SpeechCommandsV2/AMAuT/Analysis/SpeechCommandsV2_analysis-02.csv, \
./result/SpeechCommandsV2/AMAuT/Analysis/SpeechCommandsV2_analysis-03.csv, \
./result/SpeechCommandsV2/AMAuT/Analysis/SpeechCommandsV2_analysis-04.csv, \
./result/SpeechCommandsV2/AMAuT/Analysis/SpeechCommandsV2_analysis-05.csv"

python -m AuT.SpeechCommandsV2.stat_anal.stat_anal --output_file_name 'SC2-C_statistic_analysis.csv' \
    --report_list "${report_list}"