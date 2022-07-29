"""
# Split MNIST
python baselines/fromp/run_fromp.py \
    --dataset smnist \
    --n_tasks 5 \
    --batch_size 128 \
    --hidden_size 256 \
    --lr 1e-4 \
    --n_epochs 15 \
    --seed 42 \
    --n_seeds 1 \
    --n_points 40 \
    --select_method lambda_descend \
    --tau 10

# Permuted MNIST
python baselines/fromp/run_fromp.py \
    --dataset pmnist \
    --n_tasks 10 \
    --batch_size 128 \
    --hidden_size 100 \
    --lr 1e-3 \
    --n_epochs 10 \
    --seed 43 \
    --n_seeds 1 \
    --n_points 200 \
    --select_method lambda_descend \
    --tau 0.5

# CIFAR (not implemented yet)
python baselines/fromp/run_fromp.py \
    --dataset cifar \
    --n_tasks 6 \
    --batch_size 256 \
    --lr 1e-3 \
    --n_epochs 80 \
    --seed 42 \
    --n_seeds 1 \
    --n_points 200 \
    --select_method lambda_descend \
    --tau 10


python fsvi/baselines/fromp/run_fromp.py --dataset smnist_sh --n_tasks 5 --batch_size 128 --hidden_size 256 --lr 0.0001 --n_epochs 15 --seed 6 --n_seeds 1 --n_points 40 --select_method lambda_descend --tau 10.0
"""

# TODO: find a cleaner way of doing this?
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys

sys.path.insert(0, os.path.abspath(__file__ + "/../../../"))
sys.path.insert(0, os.path.abspath(__file__ + "/../../../function_space_vi/"))

import argparse
import numpy as np
import torch
from baselines.fromp.continual_learning_fromp import train
from sfsvi.general_utils.log import (
    create_logdir,
    Hyperparameters,
    set_up_logging,
    save_chkpt,
)


def add_fromp_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_tasks", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_seeds", type=int)
    parser.add_argument(
        "--n_points", type=int
    )  # number of memorable points for each task
    parser.add_argument(
        "--select_method",
        type=str,
        # choices={"lambda_ascend", "lambda_descend", "random_choice", "random_noise"}
    )
    parser.add_argument("--tau", type=float)  # should be scaled with n_points
    parser.add_argument("--use_val_split", action="store_true", default=False)
    parser.add_argument(
        "--n_permuted_tasks",
        type=int,
        default=10,
        help="The number of permuted tasks, this is only used when type of CL task is permuted tasks",
    )
    parser.add_argument(
        "--smnist_eps", type=float, default=1e-6,
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
        "--save_alt",
        action="store_true",
        default=False,
        help="Whether to save to alternative logging folder",
    )
    parser.add_argument(
        "--n_coreset_inputs_per_task",
        type=str,
        default="not_specified",
        help="Number of coreset points per task. The reason that the type is string is that the "
             "default value depends on task, but None is not accepted as integer type.",
    )
    parser.add_argument(
        "--n_steps",
        type=str,
        default="not_specified"
    )
    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="FRCL")
    add_fromp_args(parser)
    args = parser.parse_args(args)
    return args


def main(args, orig_cmd=None):
    hparams = Hyperparameters()
    hparams.from_argparse(args)
    if hparams.no_artifact:
        logger = None
        logdir = None
    else:
        logdir = create_logdir(hparams.logroot, hparams.subdir, cmd=orig_cmd)
        logger = set_up_logging(log_path=logdir / "log")
    mean_accuracies = []

    for seed in range(args.seed, (args.seed + args.n_seeds)):
        print(f"\n{args.dataset}, seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        accuracies = train(
            task=f"continual_learning_{args.dataset}",
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            lr=args.lr,
            n_epochs=args.n_epochs,
            select_method=args.select_method,
            tau=args.tau,
            use_val_split=args.use_val_split,
            n_permuted_tasks=args.n_permuted_tasks,
            seed=seed,
            smnist_eps=args.smnist_eps,
            logger=logger,
            n_coreset_inputs_per_task=hparams.n_coreset_inputs_per_task,
            n_steps=hparams.n_steps
        )
        mean_accuracies.append(np.mean(accuracies[-1]))

    split = "Validation" if args.use_val_split else "Test"
    mean = np.mean(mean_accuracies)
    std = np.std(mean_accuracies)
    print(f"\n{split} accuracy: mean = {mean:.4f}, std = {std:.4f}")
    result = {
        "mean_accuracies": mean_accuracies,
    }
    if not hparams.no_artifact:
        save_chkpt(
            p=logdir / "chkpt", hparams=hparams.as_dict(), result=result,
        )
    return logdir


def run_fromp(args):
    return main(parse_args(args), orig_cmd=["fromp"] + args)


if __name__ == "__main__":
    run_fromp(sys.argv[1:])
