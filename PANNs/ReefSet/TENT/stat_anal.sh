#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/PANNs/TENT/PAN_RS-C_TENT_analysis.csv, \
./result/ReefSet/PANNs/TENT/PAN_RS-C_TENT_analysis-02.csv, \
./result/ReefSet/PANNs/TENT/PAN_RS-C_TENT_analysis-03.csv, \
./result/ReefSet/PANNs/TENT/PAN_RS-C_TENT_analysis-04.csv, \
./result/ReefSet/PANNs/TENT/PAN_RS-C_TENT_analysis-05.csv"

python -m PANNs.ReefSet.stat_anal.stat_anal --output_path 'TENT' \
    --output_file_name 'PAN_RS-C_TENT_static_analysis.csv' \
    --report_list "${report_list}"