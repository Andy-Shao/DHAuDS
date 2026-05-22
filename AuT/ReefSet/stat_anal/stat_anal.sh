#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/AMAuT/Analysis/ReefSet_analysis.csv, \
./result/ReefSet/AMAuT/Analysis/ReefSet_analysis-02.csv, \
./result/ReefSet/AMAuT/Analysis/ReefSet_analysis-03.csv, \
./result/ReefSet/AMAuT/Analysis/ReefSet_analysis-04.csv, \
./result/ReefSet/AMAuT/Analysis/ReefSet_analysis-05.csv"

python -m AuT.ReefSet.stat_anal.stat_anal --output_file_name 'RS-C_statistic_analysis.csv' \
    --report_list "${report_list}"