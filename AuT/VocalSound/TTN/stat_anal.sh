#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/AMAuT/TTN/AuT_VS-C_TTN_analsysis.csv, \
./result/VocalSound/AMAuT/TTN/AuT_VS-C_TTN_analsysis-02.csv, \
./result/VocalSound/AMAuT/TTN/AuT_VS-C_TTN_analsysis-03.csv, \
./result/VocalSound/AMAuT/TTN/AuT_VS-C_TTN_analsysis-04.csv, \
./result/VocalSound/AMAuT/TTN/AuT_VS-C_TTN_analsysis-05.csv"

python -m AuT.VocalSound.stat_anal.stat_anal --output_file_name 'AuT_VS-C_TTN_statistic_analysis.csv' \
    --report_list "${report_list}" --output_path 'TTN'