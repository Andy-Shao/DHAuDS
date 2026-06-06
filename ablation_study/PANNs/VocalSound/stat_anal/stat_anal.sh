#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/PANNs/Ablation_study/fix-corruption_analysis.csv, \
./result/VocalSound/PANNs/Ablation_study/fix-corruption_analysis02.csv, \
./result/VocalSound/PANNs/Ablation_study/fix-corruption_analysis03.csv, \
./result/VocalSound/PANNs/Ablation_study/fix-corruption_analysis04.csv, \
./result/VocalSound/PANNs/Ablation_study/fix-corruption_analysis05.csv"

python -m ablation_study.PANNs.stat_anal.stat_anal --output_file_name 'fix-corruption_statistic_analysis.csv' \
    --report_list "${report_list}" --output_path 'Ablation_study' --dataset 'VocalSound'