import argparse
import os
import pandas as pd

from lib.utils import make_unless_exits, print_argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--report_list', type=str)
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2')
    args = ap.parse_args()
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Ablation_study')
    args.report_list = [it.strip() for it in args.report_list.split(',')]
    make_unless_exits(args.output_path)
    print_argparse(args)
    ##########################################
    records = pd.DataFrame(columns=['Dataset', 'Algorithm', 'Before-mean', 'Before-std', 'After-mean', 'After-std', 'Improve-mean', 'Improve-std'])
    for i, report_path in enumerate(args.report_list):
        report = pd.read_csv(report_path)
        if i==0: reports = [report]
        else: reports.append(report)
    unity_report = pd.concat(reports, ignore_index=True)
    B_mean = round(unity_report['before-adaptation'].mean(), ndigits=4)
    B_std = round(unity_report['before-adaptation'].std(), ndigits=4)
    A_mean = round(unity_report['after-adaptation'].mean(), ndigits=4)
    A_std = round(unity_report['after-adaptation'].std(), ndigits=4)
    I_mean = round((unity_report['after-adaptation']-unity_report['before-adaptation']).mean(), ndigits=4)
    I_std = round((unity_report['after-adaptation']-unity_report['after-adaptation']).std(), ndigits=4)
    records.loc[len(records)] = [args.dataset, args.arch, B_mean, B_std, A_mean, A_std, I_mean, I_std]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))