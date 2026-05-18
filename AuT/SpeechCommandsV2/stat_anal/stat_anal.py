import argparse
import pandas as pd
import os
from itertools import product

from lib.utils import make_unless_exits, print_argparse

def static_analyze(
        corruption_types:list[str], corruption_levels:list[str], report_ls:list[str], dataset:str, algorithm:str
    ) -> pd.DataFrame:
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Corruption-type', 'Corruption-level', 'Before-mean', 'Before-std', 'After-mean', 'After-std'])
    for i, report_addr in enumerate(report_ls):
        report = pd.read_csv(report_addr)
        if i == 0: reports = [report]
        else: reports.append(report)
    unity_report = pd.concat(reports, ignore_index=True)
    for type, level in product(corruption_types, corruption_levels):
        corruption = f'{type}-{level}'
        tmp = unity_report[unity_report['Corruption']==corruption]
        B_mean = round(tmp['Non-adapted'].mean(), ndigits=4)
        B_std = round(tmp['Non-adapted'].std(), ndigits=4)
        A_mean = round(tmp['Adapted'].mean(), ndigits=4)
        A_std = round(tmp['Adapted'].std(), ndigits=4)
        records.loc[len(records)] = [dataset, algorithm, type, level, B_mean, B_std, A_mean, A_std]

    glb_records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Corruption-type', 'Corruption-level', 'Before-mean', 'Before-std', 'After-mean', 'After-std'])
    for level in corruption_levels:
        level_records = records[records['Corruption-level']==level]
        glb_records.loc[len(glb_records)] = [
            dataset, algorithm, 'Global', level,
            round(level_records['Before-mean'].mean(), ndigits=4), round(level_records['Before-std'].mean(), ndigits=4),
            round(level_records['After-mean'].mean(), ndigits=4), round(level_records['After-std'].mean(), ndigits=4)
        ]
    return pd.concat([records, glb_records], ignore_index=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_list', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name')
    args = ap.parse_args()

    args.arch = 'AMAuT'
    args.dataset = 'SpeechCommandsV2'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    args.report_list = [it.strip() for it in args.report_list.split(',')]
    make_unless_exits(args.output_path)
    print_argparse(args)
    ##########################################
    records = static_analyze(
        corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2'],
        report_ls= args.report_list, dataset=args.dataset, algorithm=args.arch
    )
    records.to_csv(os.path.join(args.output_path, args.output_file_name))
    print('END!')