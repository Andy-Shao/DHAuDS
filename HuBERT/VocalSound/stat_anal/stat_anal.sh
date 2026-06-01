#!bin/bash
export BASE_PATH=${BASE_PATH-:'/root'}
report_list="\
./result/VocalSound/HuBERT/Analysis/VocalSound_analysis.csv, \
./result/VocalSound/HuBERT/Analysis/VocalSound_analysis-02.csv, \
./result/VocalSound/HuBERT/Analysis/VocalSound_analysis-03.csv, \
./result/VocalSound/HuBERT/Analysis/VocalSound_analysis-04.csv, \
./result/VocalSound/HuBERT/Analysis/VocalSound_analysis-05.csv"

python -m HuBERT.VocalSound.stat_anal.stat_anal --output_file_name 'HuB_VS-C_statistic_analysis.csv' \
    --report_list "${report_list}"