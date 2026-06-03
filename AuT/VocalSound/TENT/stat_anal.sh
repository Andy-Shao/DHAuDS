#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/AMAuT/TENT/AuT_VS-C_TENT_analsysis.csv, \
./result/VocalSound/AMAuT/TENT/AuT_VS-C_TENT_analsysis-02.csv, \
./result/VocalSound/AMAuT/TENT/AuT_VS-C_TENT_analsysis-03.csv, \
./result/VocalSound/AMAuT/TENT/AuT_VS-C_TENT_analsysis-04.csv, \
./result/VocalSound/AMAuT/TENT/AuT_VS-C_TENT_analsysis-05.csv"

python -m AuT.VocalSound.stat_anal.stat_anal --output_file_name 'AuT_VS-C_TENT_statistic_analysis.csv' \
    --report_list "${report_list}" --output_path 'TENT'