import argparse
import os
import pdb
import sys

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_folder)
sys.path.insert(0, os.path.join(root_folder, "function_space_vi"))
from baselines.vcl.cl_vcl import main


def add_vcl_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--n_epochs", type=int, default=100,
    )

    parser.add_argument(
        "--batch_size", type=str, default="not_specified",
    )
    parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
    parser.add_argument("--n_layers", type=int, default=2)

    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--select_method", type=str, choices={"random_choice", "k-center"}
    )
    parser.add_argument("--use_val_split", action="store_true", default=False)
    parser.add_argument(
        "--n_permuted_tasks",
        type=int,
        default=10,
        help="The number of permuted tasks, this is only used when type of CL task is permuted tasks",
    )
    parser.add_argument(
        "--logroot",
        type=str,
        help="The root result folder that store runs for this type of experiment",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        help="The subdirectory in logroot/runs/ corresponding to this run",
    )
    parser.add_argument(
        "--n_coreset_inputs_per_task",
        type=int,
        default=0,
        help="Number of coreset points per task. The reason that the type is string is that the "
        "default value depends on task, but None is not accepted as integer type.",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="If True, divide the input data by 280 to mimic the data used in the original paper",
    )
    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="VCL")
    add_vcl_args(parser)
    args = parser.parse_args(args)
    return args


def run_vcl(args):
    return main(parse_args(args), orig_cmd=["vcl"] + args)


if __name__ == "__main__":
    # import tensorflow as tf
    # print("disable tensorflow eager execution")
    # tf.compat.v1.disable_eager_execution()
    run_vcl(sys.argv[1:])
