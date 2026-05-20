#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/UrbanSound8K/AMAuT/Analysis/US8_AuT.csv, \
./result/UrbanSound8K/AMAuT/Analysis/US8_AuT-02.csv, \
./result/UrbanSound8K/AMAuT/Analysis/US8_AuT-03.csv, \
./result/UrbanSound8K/AMAuT/Analysis/US8_AuT-04.csv, \
./result/UrbanSound8K/AMAuT/Analysis/US8_AuT-05.csv"

python -m AuT.UrbanSound8K.stat_anal.stat_anal --output_file_name 'US8-C_statistic_analysis.csv' \
    --report_list "${report_list}"