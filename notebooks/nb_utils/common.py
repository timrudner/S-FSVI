"""
This file provides useful functions common to all notebooks for reproducing
results.

Note that each notebook's content is copied directly from its corresponding
python script (with the same name), because python scripts are easier to
maintain, test, and refactor.

All experiments shown in the notebooks are ran separtely on Slurm
	- sfsvi/exps/ablation/runs/reproduce_main_results_2 (for FSVI results)
	- sfsvi/exps/ablation/runs/reproduce_main_results_3 (for baselines results)
The FSVI experiments are simplified so that the size of the artifacts is small
enough for version control.
All these experiments are uploaded to GCS as well.

The stdout of the experiments is captured as artifact. These logged
experiments can be used to quickly generate cell outputs for all notebooks.

All experiments in notebooks can be tested by running their corresponding
python scripts. The main entry point for these tests is in
`publish/notebooks/main.sh`.
"""
import os
import pickle
import sys
from typing import Dict
import logging

from baselines.frcl.run_frcl_v2 import frcl_v1_to_v2
from baselines.fromp.run_fromp_v2 import fromp_v1_to_v2
from sfsvi.fsvi_utils.sfsvi_args_v2 import fsvi_v1_to_v2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)
from cli import run_config
from notebooks.nb_utils.inspect_reproduced_results import \
    display_exps_results_against_configs
from sfsvi.exps.utils import load_utils as lutils

# set this to True in order to test notebook scripts with shortened training
TEST = False
# set this to True to generate cell output of notebooks using previous logged
# experiments rather than actually running the experiments
LOAD_CACHE = False
# folder that saves the configs for reproducing various result in the paper
CONFIGS_FOLDER = os.path.join(root, "notebooks/configs")
# folder of experiments for reproducing S-FSVI results
FSVI_EXPS = os.path.join(root,
                         "notebooks/runs/reproduce_main_results_2_simplified")
FSVI_EXPS_ORIG = os.path.join(root,
                              "sfsvi/exps/ablation/runs/reproduce_main_results_2")
# folder of experiments for reproducing baselines results (e.g. FRCL, FROMP, VCL)
BASELINES_EXPS = os.path.join(root, "notebooks/runs/reproduce_main_results_3")


def display_results_of_cached_exp():
    """For visually inspect the performance of the experiments"""
    exps = lutils.read_folder(FSVI_EXPS, only_config=True, only_successful=True)
    display_exps_results_against_configs(exps)

    exps = lutils.read_folder(BASELINES_EXPS, only_config=True,
                              only_successful=True)
    display_exps_results_against_configs(exps, baseline=True)


def simplify_and_upload_experiments():
    """Run this function after running the experiments."""
    lutils.simplify_subdir(FSVI_EXPS_ORIG, new_subdir=FSVI_EXPS)
    lutils.upload_subdir(FSVI_EXPS, dry_run=False)
    lutils.upload_subdir(BASELINES_EXPS, dry_run=False)


def v1_to_v2(config, runner):
    if runner == "frcl":
        return frcl_v1_to_v2(config)
    elif runner == "fromp":
        return fromp_v1_to_v2(config)
    elif runner == "fsvi":
        return fsvi_v1_to_v2(config)
    else:
        return config


def read_config_and_run(
    filename: str,
    task_sequence: str,
    runner: str = "fsvi",
    load_chkpt: bool = False,
) -> str:
    """
	Read a config and run the experiment.

	:param filename: name of pickle file containing the experiment configs.
	:param task_sequence: the name of the task sequence.
	:param runner: the continual learning method, e.g. fsvi, frcl.
	:param load_chkpt: if True, load an existing checkpoint instead of training
		the model from scratch.
	:return:
		a path to the directory containing the artifact of the experiment.
	"""
    path = os.path.join(CONFIGS_FOLDER, filename)
    with open(path, "rb") as p:
        configs = pickle.load(p)
    config = configs[task_sequence]
    if LOAD_CACHE or load_chkpt:
        print("Loading from cache:")
        exp = load_experiment(config, _get_cache_folder(runner))
        if exp:
            return exp["path"]
        print("no cache available, running...")
    config = v1_to_v2(config, runner=runner)
    if TEST:
        simplify_config(config)
    logdir = run_config(config, runner=runner)
    return logdir


def _get_cache_folder(runner: str):
    if runner == "fsvi":
        return FSVI_EXPS
    else:
        return BASELINES_EXPS


def show_final_average_accuracy(exp, runner: str = "fsvi"):
    if runner == "fsvi":
        training_log_df = exp['chkpt']['training_log_df']
        cols = [c for c in training_log_df if c[:8] == "test_acc"]
        print(training_log_df[cols].iloc[-1].values.mean())
    elif runner == "frcl" or runner == "fromp":
        if "training_log_df" in exp["chkpt"]:
            show_final_average_accuracy(exp)
        else:
            print(exp['chkpt']['result']['mean_accuracies'][0])
    elif runner == "vcl":
        print(exp['chkpt']['result'][-1].mean())
    else:
        raise NotImplementedError(runner)


def simplify_config(config):
    config["subdir"] = "test"
    if "n_permuted_tasks" in config:
        config["n_permuted_tasks"] = 2
    if "fsvi" in config.get("model_type", ""):
        config["epochs"] = 1
        if "mnist" in config["data_training"]:
            config["architecture"] = "fc_2_2"
        if "omniglot" in config["data_training"]:
            config["n_omniglot_tasks"] = 2
        if "epochs_first_task" in config:
            config["epochs_first_task"] = 2
        config["debug_n_train"] = 300
    else:
        if "n_epochs" in config:
            config["n_epochs"] = 1
        if "n_iterations_train" in config:
            config["n_iterations_train"] = 2
        config["hidden_size"] = 2


def load_experiment(config: Dict, exp_folder: str):
    """Load experiment that was logged instead of actually running an
	experiment."""
    exps = lutils.read_folder(exp_folder, only_config=True,
                              only_successful=True)
    keys = set(lutils.find_diff_keys(exps))
    config = {k: v for k, v in config.items() if
              k in keys and k not in ["seed", "subdir", "data_ood"]}
    matched_exps = lutils.filter_exps(exps, d=config)
    if not matched_exps:
        print(f"No experiment found in {exp_folder}")
        return
    exp = matched_exps[0]
    # print stdout of the experiment which contains training dynamics
    log = lutils.read_slurm_log(exp)
    clean_log = lutils.remove_warnings_from_stdout(log)
    print(clean_log)
    return exp
