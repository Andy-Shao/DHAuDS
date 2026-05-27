#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/AMAuT/Analysis/VocalSound_analysis.csv, \
./result/VocalSound/AMAuT/Analysis/VocalSound_analysis-02.csv, \
./result/VocalSound/AMAuT/Analysis/VocalSound_analysis-03.csv, \
./result/VocalSound/AMAuT/Analysis/VocalSound_analysis-04.csv, \
./result/VocalSound/AMAuT/Analysis/VocalSound_analysis-05.csv"

python -m AuT.VocalSound.stat_anal.stat_anal --output_file_name 'VS-C_statistic_analysis.csv' \
    --report_list "${report_list}"