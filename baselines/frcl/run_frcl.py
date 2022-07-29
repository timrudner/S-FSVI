# TODO: find a cleaner way of doing this?
import os
import pdb
import sys

sys.path.insert(0, os.path.abspath(__file__ + "/../../../"))
sys.path.insert(0, os.path.abspath(__file__ + "/../../../function_space_vi/"))

import argparse
import numpy as np
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# import jax
# jax.config.update('jax_platform_name', 'cpu')
from baselines.frcl.continual_learning_frcl import train
from sfsvi.general_utils.log import (
    create_logdir,
    Hyperparameters,
    set_up_logging,
    save_chkpt,
)


def add_frcl_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_iterations_train", type=int)
    parser.add_argument("--n_iterations_discr_search", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument(
        "--select_method",
        type=str,
        # choices={"random_choice", "random_noise", "trace_term"},
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
        "--n_omniglot_inducing_chars",
        type=int,
        default=2,
        help="the number of inducing inputs per character, FRCL paper reported "
             "results on 1, 2, 3"
    )
    parser.add_argument(
        "--n_omniglot_tasks",
        type=int,
        default=50,
        help="the number of omniglot tasks, must be not greater than 50 (FRCL paper used 50 tasks)"
    )
    parser.add_argument(
        "--randomize_test_split",
        action="store_true",
        default=False,
        help="If True, randomize test split using `seed`"
    )
    parser.add_argument(
        "--randomize_task_sequence",
        action="store_true",
        default=False,
        help="If True, randomize task sequence using `seed`"
    )
    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="FRCL")
    add_frcl_args(parser)
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
    for seed in range(hparams.seed, (hparams.seed + hparams.n_seeds)):
        print(f"\n{hparams.dataset}, seed {seed}")
        tf.random.set_seed(seed)
        accuracies = train(
            task=f"continual_learning_{hparams.dataset}",
            batch_size=hparams.batch_size,
            hidden_sizes=[hparams.hidden_size] * hparams.n_layers,
            learning_rate=hparams.learning_rate,
            n_iterations_train=hparams.n_iterations_train,
            n_iterations_discr_search=hparams.n_iterations_discr_search,
            select_method=hparams.select_method,
            use_val_split=hparams.use_val_split,
            seed=seed,
            n_permuted_tasks=hparams.n_permuted_tasks,
            logger=logger,
            n_coreset_inputs_per_task=hparams.n_coreset_inputs_per_task,
            n_omniglot_inducing_chars=hparams.n_omniglot_inducing_chars,
            n_omniglot_tasks=hparams.n_omniglot_tasks,
            randomize_test_split=hparams.randomize_test_split,
            randomize_task_sequence=hparams.randomize_task_sequence,
        )
        mean_accuracies.append(np.mean(accuracies[-1]))

    split = "Validation" if hparams.use_val_split else "Test"
    mean = np.mean(mean_accuracies)
    std = np.std(mean_accuracies)
    print(f"\n{split} accuracy: mean = {mean:.4f}, std = {std:.4f}")

    result = {
        "mean_accuracies": mean_accuracies,
    }
    if hparams.save_alt and not hparams.no_artifact:
        save_chkpt(
            p=logdir / "chkpt", hparams=hparams.as_dict(), result=result,
        )
    return logdir


def run_frcl(args):
    return main(parse_args(args), orig_cmd=["frcl"] + args)


if __name__ == "__main__":
    run_frcl(sys.argv[1:])
