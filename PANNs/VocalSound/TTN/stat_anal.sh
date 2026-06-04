#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/PANNs/TTN/PAN_VS-C_TTN_analysis.csv, \
./result/VocalSound/PANNs/TTN/PAN_VS-C_TTN_analysis-02.csv, \
./result/VocalSound/PANNs/TTN/PAN_VS-C_TTN_analysis-03.csv, \
./result/VocalSound/PANNs/TTN/PAN_VS-C_TTN_analysis-04.csv, \
./result/VocalSound/PANNs/TTN/PAN_VS-C_TTN_analysis-05.csv"

python -m PANNs.VocalSound.stat_anal.stat_anal --output_path 'TTN' \
    --output_file_name 'PAN_VS-C_TTN_static_analysis.csv' \
    --report_list "${report_list}"