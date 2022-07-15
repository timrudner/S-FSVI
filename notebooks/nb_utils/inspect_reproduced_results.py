"""
This file is for analysing the experiments to see if we can reproduce the
accuracies of our method on MNIST in the paper.
"""
import os
import pdb
import pickle

import numpy as np

import sfsvi.exps.utils.load_utils as lutils
from sfsvi.general_utils.log import PROJECT_ROOT

FSVI_CONFIGS = [
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_match.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_optimized.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_no_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_minimal_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_cifar.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fsvi_omniglot.pkl"),
]

BASELINES_CONFIGS = [
    os.path.join(PROJECT_ROOT, "notebooks/configs/frcl_no_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/frcl_with_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fromp_no_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/fromp_with_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/vcl_no_coreset.pkl"),
    os.path.join(PROJECT_ROOT, "notebooks/configs/vcl_random_coreset.pkl"),
]


def _load_configs(paths):
    configs = {}
    for p in paths:
        name = p.split("/")[-1].rstrip(".pkl")
        print(name)
        with open(p, "rb") as f:
            configs[name] = pickle.load(f)
    return configs


def select_exps(exps, configs):
    res = {}
    keys = set(lutils.find_diff_keys(exps))
    for group in configs:
        temp = {}
        for ts in configs[group]:
            c = configs[group][ts]
            c = {
                k: v
                for k, v in c.items()
                if k in keys and k not in ["seed", "subdir", "data_ood"]
            }
            temp[ts] = lutils.filter_exps(exps, d=c)
            if not temp[ts]:
                print(f"No experiment found for config:\n\t{c}\n")
                close_exps = lutils.filter_exps(exps, d={'data_training': c['data_training']})
                print(lutils.find_diff_keys(close_exps + [{'config': c}]))
                import pdb; pdb.set_trace()
            # pdb.set_trace()
        res[group] = temp
    return res


def get_test_callable(group: str):
    if "fsvi" in group:
        return lutils.get_final_avg_test_accuracy
    elif "fromp" in group:
        return lutils.get_final_average_test_accuracy_fromp
    elif "frcl" in group:
        return lambda exp: np.mean(lutils.get_average_test_accuracy_per_task_frcl(exp))
    elif "vcl" in group:
        return lambda exp: np.mean(lutils.get_average_test_accuracy_per_task_vcl(exp))
    else:
        raise NotImplementedError(group)


def show_reprod_res(gexps):
    for group in gexps:
        print("*" * 80)
        print(group)
        for ts in gexps[group]:
            if not gexps[group][ts]:
                print(f"\texperiment {group}, {ts} is not found")
                continue
            df = lutils.report_test_accuracies(
                gexps[group][ts], test_func=get_test_callable(group)
            )
            if len(df) != 1 and df.ndim != 1:
                pdb.set_trace()
            mean, stderr = df["mean"].iloc[0], df["standard_error"].iloc[0]
            print(f"\t{ts}\t\t{mean}+-{stderr}")


def display_exps_results_against_configs(exps, baseline=False):
    paths = BASELINES_CONFIGS if baseline else FSVI_CONFIGS
    configs = _load_configs(paths)
    gexps = select_exps(exps, configs)
    show_reprod_res(gexps)


def load_all_exps(baseline=False):
    folder = (
        "reproduce_main_results_3"
        if baseline
        else "reproduce_main_results_2_simplified"
    )
    path = os.path.join(PROJECT_ROOT, "notebooks/runs", folder)
    print(f"Reading from {path}")
    exps = lutils.read_folder(path, only_config=True, only_successful=True)
    return exps
