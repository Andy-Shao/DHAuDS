#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/PANNs/TTN/PAN_RS-C_TTN_analysis.csv, \
./result/ReefSet/PANNs/TTN/PAN_RS-C_TTN_analysis-02.csv, \
./result/ReefSet/PANNs/TTN/PAN_RS-C_TTN_analysis-03.csv, \
./result/ReefSet/PANNs/TTN/PAN_RS-C_TTN_analysis-04.csv, \
./result/ReefSet/PANNs/TTN/PAN_RS-C_TTN_analysis-05.csv"

python -m PANNs.ReefSet.stat_anal.stat_anal --output_path 'TTN' \
    --output_file_name 'PAN_RS-C_TTN_static_analysis.csv' \
    --report_list "${report_list}"