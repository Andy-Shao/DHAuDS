#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/UrbanSound8K/PANNs/TTN/PAN_US8-C_TTN_analysis.csv, \
./result/UrbanSound8K/PANNs/TTN/PAN_US8-C_TTN_analysis-02.csv, \
./result/UrbanSound8K/PANNs/TTN/PAN_US8-C_TTN_analysis-03.csv, \
./result/UrbanSound8K/PANNs/TTN/PAN_US8-C_TTN_analysis-04.csv, \
./result/UrbanSound8K/PANNs/TTN/PAN_US8-C_TTN_analysis-05.csv"

python -m PANNs.UrbanSound8K.stat_anal.stat_anal --output_file_name 'PAN_US8-C_TTN_statistic_analysis.csv' \
    --report_list "${report_list}" --output_path 'TTN'