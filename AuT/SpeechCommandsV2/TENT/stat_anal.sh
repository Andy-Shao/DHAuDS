export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/SpeechCommandsV2/AMAuT/TENT/AuT_SC2-C_TENT_analysis.csv, \
./result/SpeechCommandsV2/AMAuT/TENT/AuT_SC2-C_TENT_analysis-02.csv, \
./result/SpeechCommandsV2/AMAuT/TENT/AuT_SC2-C_TENT_analysis-03.csv, \
./result/SpeechCommandsV2/AMAuT/TENT/AuT_SC2-C_TENT_analysis-04.csv, \
./result/SpeechCommandsV2/AMAuT/TENT/AuT_SC2-C_TENT_analysis-05.csv"

python -m AuT.SpeechCommandsV2.stat_anal.stat_anal --output_path 'TENT' \
    --output_file_name 'AuT_SC2-C_TENT_static_analysis.csv' \
    --report_list "${report_list}"