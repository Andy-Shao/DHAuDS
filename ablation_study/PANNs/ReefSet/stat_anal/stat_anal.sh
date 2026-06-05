#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/PANNs/Ablation_study/fix-corruption_analysis.csv, \
./result/ReefSet/PANNs/Ablation_study/fix-corruption_analysis02.csv, \
./result/ReefSet/PANNs/Ablation_study/fix-corruption_analysis03.csv, \
./result/ReefSet/PANNs/Ablation_study/fix-corruption_analysis04.csv, \
./result/ReefSet/PANNs/Ablation_study/fix-corruption_analysis05.csv"

python -m ablation_study.PANNs.ReefSet.stat_anal.stat_anal --output_file_name 'fix-corruption_statistic_analysis.csv' \
    --report_list "${report_list}" --output_path 'Ablation_study'