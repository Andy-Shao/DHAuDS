#!bin/bash
export BASE_PATH=${BASE_PATH-:'/root'}
report_list="\
./result/ReefSet/HuBERT/Ablation_study/fix-corruption_analysis.csv, \
./result/ReefSet/HuBERT/Ablation_study/fix-corruption_analysis02.csv, \
./result/ReefSet/HuBERT/Ablation_study/fix-corruption_analysis03.csv, \
./result/ReefSet/HuBERT/Ablation_study/fix-corruption_analysis04.csv, \
./result/ReefSet/HuBERT/Ablation_study/fix-corruption_analysis05.csv"

python -m ablation_study.HuBERT.stat_anal.stat_anal --output_path './result' \
    --output_file_name 'stat_anal.csv' --report_list "${report_list}" --dataset 'ReefSet'