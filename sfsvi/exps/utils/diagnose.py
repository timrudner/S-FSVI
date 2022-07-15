"""Command line utils for summarising the success and failures of experiment
runs."""
import argparse
import os
import sys

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_folder)
sys.path.insert(0, os.path.join(root_folder, "function_space_vi"))
from sfsvi.general_utils.log import EXPS_ROOT
from sfsvi.exps.utils import load_utils as lutils


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose")
    parser.add_argument("-r", "--runs", type=str, required=True)
    parser.add_argument("--log", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    run_folder_path = EXPS_ROOT / args.runs
    lutils.diagnose_exp_folder(run_folder_path, log=args.log)


if __name__ == "__main__":
    main(parse_args())
