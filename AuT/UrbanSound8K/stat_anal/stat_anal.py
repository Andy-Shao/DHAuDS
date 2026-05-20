import argparse
import os

from lib.utils import make_unless_exits, print_argparse
from AuT.SpeechCommandsV2.stat_anal.stat_anal import static_analyze

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_list', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name')
    args = ap.parse_args()

    args.arch = 'AMAuT'
    args.dataset = 'UrbanSound8K'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    args.report_list = [it.strip() for it in args.report_list.split(',')]
    make_unless_exits(args.output_path)
    print_argparse(args)
    ##########################################
    records = static_analyze(
        corruption_types=['WHN', 'ENSC', 'PSH', 'TST'], corruption_levels=['L1', 'L2'],
        report_ls= args.report_list, dataset=args.dataset, algorithm=args.arch
    )
    records.to_csv(os.path.join(args.output_path, args.output_file_name))
    print('END!')