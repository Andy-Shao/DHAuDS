#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/AMAuT/TTN/AuT_RS-C_TTN_analysis.csv, \
./result/ReefSet/AMAuT/TTN/AuT_RS-C_TTN_analysis-02.csv, \
./result/ReefSet/AMAuT/TTN/AuT_RS-C_TTN_analysis-03.csv, \
./result/ReefSet/AMAuT/TTN/AuT_RS-C_TTN_analysis-04.csv, \
./result/ReefSet/AMAuT/TTN/AuT_RS-C_TTN_analysis-05.csv"

python -m AuT.ReefSet.stat_anal.stat_anal --output_path 'TTN' \
    --output_file_name 'AuT_RS-C_TTN_static_analysis.csv' \
    --report_list "${report_list}" --output_path 'TTN'