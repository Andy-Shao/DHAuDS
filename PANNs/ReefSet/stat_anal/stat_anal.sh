#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/PANNs/Analysis/PAN_RS-C_analysis.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_analysis-02.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_analysis-03.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_analysis-04.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_analysis-05.csv"

python -m PANNs.ReefSet.stat_anal.stat_anal \
    --output_file_name 'PAN_RS-C_static_analysis.csv' \
    --report_list "${report_list}"