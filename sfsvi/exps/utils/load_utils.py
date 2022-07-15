"""Utilities to load, analyse experiments."""
import argparse
import json
import os
import pdb
import pickle
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sfsvi.exps.utils.config_template import ConfigTemplate
from sfsvi.exps.utils.configs import CL_TEMPLATE
from sfsvi.exps.utils.configs import CL_TEMPLATE_v2
from sfsvi.exps.utils.configs import EXPS_ROOT
from sfsvi.exps.utils.generate_cmds import ENTRY_POINT
from sfsvi.general_utils.log import PROJECT_ROOT
from sfsvi.general_utils.log import save_chkpt

try:
    from sfsvi.exps.utils.baselines_configs import (
        FRCL_TEMPLATE,
        FROMP_TEMPLATE,
        VCL_TEMPLATE,
    )
except:
    print(
        "---------------- Could not load either VCL, FRCL, or FROMP! ----------------"
    )

DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON = (
    "subdir",
    "logroot",
    "plotting",
    "figsize",
    "debug",
    "save_path",
)


FIG_ROOT = EXPS_ROOT / "figures"
os.makedirs(FIG_ROOT, exist_ok=True)


def is_valid_run_folder(run_folder_path: str) -> bool:
    """
    If the run folder contains a chkpt file
    """
    return os.path.exists(os.path.join(run_folder_path, "chkpt"))


def load_chkpt(chkpt: Path):
    try:
        with Path(chkpt).open("rb") as f:
            chkpt = pickle.load(f)
        return chkpt

    except FileNotFoundError:
        print("checkpoint not found")
        return {}


def select_template(prefix: str) -> ConfigTemplate:
    if "frcl" in prefix:
        template = FRCL_TEMPLATE
    elif "fromp" in prefix:
        template = FROMP_TEMPLATE
    elif "vcl" in prefix:
        template = VCL_TEMPLATE
    elif "cl_v2" in prefix:
        template = CL_TEMPLATE_v2
    elif "cl" in prefix:
        template = CL_TEMPLATE
    else:
        raise ValueError(f"Unrecognized prefix {prefix}")
    return template


def cmd_to_dict(cmd_line_str: str) -> Dict:
    """
    example input:
        python /nfs-share/qixuan/miniconda3/envs/fsvi-cl/bin/fsvicl --data_training continual_learning_smnist
    example output:
        {'arch': 'lenet5_caffe', 'dataset': 'mnist', 'epochs': 30, 'preprune': 'snip', 'keep-ratio': 0.95,
    'command': 'train'}
    """
    cmd = cmd_line_str.split(" ")
    prefixes = []
    for i, x in enumerate(cmd):
        if x[:2] == "--":
            break
        else:
            prefixes.append(x)
    cmd_str = " ".join(cmd[i:])
    template = select_template(prefix=" ".join(prefixes))
    return template.parse_args(cmd_str=cmd_str, ignore_unknown=True)


def read_config_from_command_line(exp_path: str, file_name="command_line.txt") -> Dict:
    """
    :param exp_path: e.g. ~/gp_prune/runs/2020-12-05-Saturday-23-07-28
    :param file_name: name file storing the command line within the exp_path folder
    :return:
    """
    with open(os.path.join(exp_path, file_name), "r") as f:
        cmd = f.read()
    return cmd_to_dict(cmd)


def read_command_line(e: Dict, file_name="command_line.txt") -> str:
    """
    :param exp_path: e.g. ~/gp_prune/runs/2020-12-05-Saturday-23-07-28
    :param file_name: name file storing the command line within the exp_path folder
    :return:
    """
    with open(os.path.join(e["path"], file_name), "r") as f:
        cmd = f.read()
    return cmd


def read_configs_from_job_file(job_file: str) -> List[Dict]:
    """
    :param exp_path: e.g. ~/gp_prune/runs/2020-12-05-Saturday-23-07-28
    :param file_name: name file storing the command line within the exp_path folder
    :return:
    """
    with open(job_file, "r") as f:
        cmds = f.readlines()
    return [cmd_to_dict(cmd) for cmd in cmds]


def read_config_from_hparams(exp_path: str) -> Dict:
    """
    Alternative way of loading config.

    Instead of reading config from command_line.txt, load it directly from hparams.

    TODO: use this as default way of loading experiment config.

    :param exp_path: path to the experiment run.
    :return: loaded experiment config.
    """
    with Path(Path(exp_path) / "chkpt").open("rb") as f:
        state = pickle.load(f)
    return state["hparams"]


def read_folder(
    exps_folder: str,
    only_successful: bool = False,
    only_config: bool = False,
    only_failed: bool = False,
    d: Dict = {},
    **kwargs,
) -> List[Dict]:
    """
    Read a folder of experiment runs.

    :param exps_folder: the path to the folder that contains runs of experiments
    :param only_successful: if True, only load successful experiments
    :param only_config: if True, only load the experiment config
    :param d: if provided, it filter the experiments to load
        for example, d = {'postprune': 'rev_kl'} will only load experiments with rev_kl as postprune method
    :param kwargs: kwargs passed to is_config_accepted
    :return:
    """
    exps = [os.path.join(exps_folder, e) for e in os.listdir(exps_folder)]
    results = []
    for e in tqdm(exps, desc="loading experiments"):
        temp_e = read_exp(e, only_config=True)
        if only_successful and not is_valid_run_folder(e):
            continue
        if only_failed and is_valid_run_folder(e):
            continue
        if not d or is_config_accepted(temp_e["config"], d, **kwargs):
            if not only_config:
                results.append(read_exp(e, only_config=False))
            else:
                results.append(temp_e)
    return results


def get_latest_exp(folder):
    exps = read_folder(folder, only_config=True)
    return max(exps, key=lambda e: exp_datetime(e))


def delete_experiments(exps: List[Dict], only_failed=True):
    for e in exps:
        if not only_failed or not is_valid_run_folder(e["path"]):
            shutil.rmtree(e["path"])
            print(f"Removing folder {e['path']}")


def load_chkpt_by_keys(chkpt_path: str, exclude: List[str] = []) -> Dict:
    """
    :param chkpt_path: path to a chkpt_final_trained_model file
    :param exclude: the keys to exclude during loading
    :return:
    """
    with Path(chkpt_path).open("rb") as f:
        chkpt = pickle.load(f)
    return {k: chkpt[k] for k in chkpt.keys() if k not in exclude}


def read_exp(exp_path: str, only_config=False) -> Dict:
    """
    Load an experiment run folder.

    :param exp_path: path to an experiment run
    :param only_config: if True, only load experiment config
    :return:
    """
    d = {"path": os.path.abspath(exp_path)}
    config = read_config_from_command_line(exp_path)
    d["config"] = config
    if only_config:
        return d
    chkpt_path = Path(exp_path) / "chkpt"
    d["chkpt"] = {**load_chkpt(chkpt_path)}
    return d


def load_exp(e: Dict, overwrite=False) -> Dict:
    if overwrite or "chkpt" not in e:
        loaded_exp = read_exp(e["path"], only_config=False)
        e.update(loaded_exp)
    return e


def load_exps(exps: List[Dict], overwrite=False, verbose=True) -> List[Dict]:
    return [
        load_exp(e, overwrite=overwrite)
        for e in tqdm(exps, desc="loading chkpt", disable=not verbose)
    ]


def read_log(exp: Dict, log_file="log") -> str:
    try:
        with open(Path(exp["path"]) / log_file, "r") as f:
            log = f.read()
    except Exception as exp:
        print(f"encountered exception: {exp}")
        log = ""
    return log


def find_slurm_out_file(exp_folder: str):
    prog = re.compile("^slurm-(?P<job_id>\d+)_(?P<task_id>\d+)\.out$")
    for file in os.listdir(exp_folder):
        match = prog.match(file)
        if match:
            return file
    print(f"no slurm output file found in {exp_folder}")


def has_slurm_file(exp: Dict) -> bool:
    slurm_file = find_slurm_out_file(exp["path"])
    return slurm_file is not None


def read_slurm_log(exp: Dict):
    slurm_file = find_slurm_out_file(exp["path"])
    if slurm_file:
        return read_log(exp, log_file=slurm_file)
    else:
        return ""


def d1_include_d2(d1: Dict, d2: Dict) -> bool:
    return all([d1.get(k) == d2.get(k) for k in d2])


def read_extra(exps_folder: str, existing_exps: List[Dict], **kwargs):
    """
    Read experiments that are not in existing_exps.

    :param exps_folder: the path to the folder containing several experiment runs
    :param existing_exps: the experiments that are already loaded
    """
    exps = read_folder(exps_folder, **kwargs)
    exps = [e for e in exps if e["loaded"]["params"] is not None]
    for e in exps:
        chkpt = e["config"]["chkpt"]
        e["config"]["chkpt"] = os.path.basename(os.path.dirname(chkpt))

    new_exps = [
        e
        for e in exps
        if all([not d1_include_d2(e2["config"], e["config"]) for e2 in existing_exps])
    ]
    return new_exps


def deduplicate(exps: List[Dict]) -> List[Dict]:
    new_exps = []
    for e in exps:
        if all([not d1_include_d2(e2["config"], e["config"]) for e2 in new_exps]):
            new_exps.append(e)
    return new_exps


def subtract(exps_1: List[Dict], exps_2: List[Dict]) -> List[Dict]:
    exps_1_rest = []
    for e in exps_1:
        if len(deduplicate_v2(exps_2 + [e])) != len(exps_2):
            exps_1_rest.append(e)
    return exps_1_rest


def deduplicate_v2(
    exps: List[Dict], keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON
) -> List[Dict]:
    if not exps:
        return exps
    keys_to_consider = list(
        filter(lambda k: k not in keys_to_ignore, list(exps[0]["config"].keys()))
    )
    make_immutable_config = get_make_immutable_config(keys=keys_to_consider)
    immutable_configs = set()
    deduplicated_exps = []
    for e in exps:
        immutable_config = make_immutable_config(e)
        if immutable_config not in immutable_configs:
            immutable_configs.add(immutable_config)
            deduplicated_exps.append(e)
    print(
        f"Removed {len(exps) - len(deduplicated_exps)} duplicated experiments\n"
        f"Left {len(deduplicated_exps)} experiments"
    )
    return deduplicated_exps


def diagnose_exp_folder(exps_folder: str, log=False):
    """
    Show the failed runs and their logs.

    :param exps_folder: path to a folder containing experiment runs
    :return:
    """
    cexps = read_folder(exps_folder, only_config=True)
    succ_exps = [e for e in cexps if is_valid_run_folder(e["path"])]
    empty_cexps = [e for e in cexps if not is_valid_run_folder(e["path"])]
    print(f"{len(cexps)} runs found in the folder")
    print(f"{len(empty_cexps)} runs failed, displaying their config and logs...")
    summarise_causes(empty_cexps, succ_exps)
    if log:
        keys = find_diff_keys(empty_cexps)
        show_exps(empty_cexps, keys=keys, log=True)


def partition_by_cond(elements: Sequence, cond: Callable) -> Tuple[List, List]:
    yes = [e for e in elements if cond(e)]
    no = [e for e in elements if not cond(e)]
    return yes, no


def summarise_causes(exps: List[Dict], succ_exps):
    exps, no_slurm_exps = partition_by_cond(exps, has_slurm_file)
    print(f"There are {len(no_slurm_exps)} exps that don't have slurm logs")
    oom_exps, exps = partition_by_cond(exps, is_oom)
    cpu_exps, _ = partition_by_cond(exps, back_to_cpu)
    print(
        f"There are {len(oom_exps)} exps that got OOM error \n"
        f"amoung which {len(cpu_exps)} exps fall back to CPU"
    )
    print(f"There are {len(exps)} exps with unknown cause")
    if oom_exps:
        print(f"Here is a quick summary of the OOM experiments:")
        show_stats(oom_exps, add_keys=["architecture", "coreset", "n_inducing_inputs"])
        show_nodes_stats(oom_exps)
        print(f"Here is a quick summary of the successful experiments:")
        show_stats(succ_exps, add_keys=["architecture", "coreset", "n_inducing_inputs"])
        show_nodes_stats(succ_exps)
    # analyse_nodes_for_oom(succ_exps=succ_exps, oom_exps=oom_exps)
    if exps:
        print("-" * 100)
        print("Here is the logs of unknown experiments")
        show_exps(exps, keys=find_diff_keys(exps), log=True)


def analyse_nodes_for_oom(succ_exps, oom_exps):
    print("-" * 100)
    print("Here are nodes producing oom_exps")
    show_nodes_stats(oom_exps)
    print("Here are nodes of exps that are similar but successful")
    similar_succ_exps = find_similar_subsets(succ_exps, oom_exps)
    show_nodes_stats(similar_succ_exps)


def find_similar_subsets(exps: List[Dict], subset: List[Dict]) -> List[Dict]:
    if not subset:
        return []
    e = subset[0]
    common_keys = (
        set(e["config"].keys())
        - set(find_diff_keys(subset))
        - set(DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON)
    )
    d = {k: e["config"][k] for k in common_keys}
    similar_exps = filter_exps(exps, d=d)
    return similar_exps


def find_similar_exps(exps: List[Dict], e: Dict, keys_to_ignore: List[str]) -> List[Dict]:
    keys = sorted(
        set(e["config"].keys()) - set(DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON) - set(keys_to_ignore)
    )
    d = {k: e["config"][k] for k in keys}
    similar_exps = filter_exps(exps, d=d)
    return similar_exps


def show_nodes_stats(exps: List[Dict]):
    nodes = [get_node(e) for e in exps]
    print("-" * 100)
    print("nodes used are")
    print(pd.Series(nodes).value_counts())


def show_exps(exps: List[Dict], keys: List[str] = None, log: bool = False):
    """
    Show config and log information of loaded experiments

    :param exps: list of loaded experiments
    :param keys: keys of experiments config to include
    :param log: show log or not
    :return:
    """
    for e in exps:
        print("-" * 100)
        if keys is None:
            config = e["config"]
        else:
            config = {k: e["config"][k] for k in keys if k in e["config"]}
        print(json.dumps(config, sort_keys=True, indent=4))

        if log:
            print("-" * 100)
            print("log file")
            print(read_log(e))
            print("-" * 100)
            print("slurm file")
            print(read_slurm_log(e))


def show_stats(exps, keys: List[str] = None, add_keys: List[str] = None):
    keys = find_diff_keys(exps) if keys is None else keys
    keys.extend([] if add_keys is None else add_keys)
    keys = sorted(set(keys))
    for k in keys:
        print("-" * 100)
        print("key:", k)
        print(value_counts(exps, k))


def value_counts(exps: List[Dict], key: str):
    configs = pd.Series([to_int_if_possible(e["config"].get(key)) for e in exps])
    return configs.value_counts()


def find_diff_keys(exps: List[Dict]) -> List[str]:
    """
    Find a list of config keys that these experiments are different from each other.

    :param exps: a list of loaded experiments.
    :return: a list of config keys on which the experiments are different from each other.
    """
    if len(exps) <= 1:
        return []
    configs = [e["config"] for e in exps]
    c0 = configs[0]
    keys_of_diff = []
    for k in c0:
        vals = set(tuple([if_list_to_tuple(c.get(k)) for c in configs]))
        if len(vals) > 1:
            keys_of_diff.append(k)
    return keys_of_diff


def if_list_to_tuple(x):
    return tuple(x) if isinstance(x, list) else x


def filter_exps(exps, d, **condition) -> List[Dict]:
    return list(filter(lambda e: is_config_accepted(e["config"], d, **condition), exps))


def is_config_accepted(
    config: Dict, d: Dict, union: bool = False, negate: bool = False
) -> bool:
    """
    Check whether the experiment config matches [d].

    :param config: a config dictionary for a run of pruning
    :param d: checks whether config matches d
        for example, if d is {'postprune': 'rev_kl'}, then it checks whether the config dictionary has this
        key-value pair
    :param union: if True, the config is accepted if at least one of the key-value pairs in d is satisifed
            otherwise, the config is accepted only if all the key-value pairs in d are satisifed.
    :param negate: if True, the config is accepted if it matches [d]; otherwise, the config is accepted
            if it doesn't match [d]
    :return:
    """
    d = {
        k: v if isinstance(v, list) and not isinstance(config.get(k), list) else [v]
        for k, v in d.items()
    }
    d = {k: [to_numeric_if_possible(x) for x in v] for k, v in d.items()}
    conds = [to_numeric_if_possible(config.get(k)) in v for k, v in d.items()]
    cond = any(conds) if union else all(conds)
    if negate:
        cond = not cond
    return cond


def append_result(exp: Dict, key, value, overwrite=False):
    """
    Save a key value pair to the chkpt of the experiment

    :param exp: the experiment
    :param key:
    :param value:
    :return:
    """
    path = os.path.join(exp["path"], "chkpt_final_trained_model")
    with Path(path).open("rb") as f:
        chkpt = pickle.load(f)
    if not overwrite:
        assert key not in chkpt, (
            f"chkpt has already {key}, the original value is {chkpt[key]}, "
            f"the one to update is {value}"
        )
    chkpt[key] = value
    with Path(path).open("wb") as f:
        pickle.dump(chkpt, f)


def find_pairs_with_nb_diff(exps, nb_diff=1):
    pairs = []
    for i in range(len(exps) - 1):
        for j in range(i + 1, len(exps)):
            keys = find_diff_keys([exps[i], exps[j]])
            if len(keys) == nb_diff:
                pairs.append([i, j])
    return pairs


def to_int_if_possible(x):
    if isinstance(x, float):
        return x
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def to_numeric_if_possible(x):
    if isinstance(x, float):
        return x
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return float(x)
        except (ValueError, TypeError):
            return x


def partition_exps(
    exps: List[Dict], keys_to_ignore: Tuple[str, ...] = ()
) -> List[List[int]]:
    keys = find_diff_keys(exps)
    keys = list(filter(lambda x: x not in keys_to_ignore, keys))
    make_immutable_config = get_make_immutable_config(keys)
    configs = [make_immutable_config(e) for e in exps]
    partitions = defaultdict(list)
    for i, c in enumerate(configs):
        partitions[c].append(i)
    return list(partitions.values())


def get_make_immutable_config(keys):
    def _make_immutable_config(e):
        return make_immutable_sub_spec(e, keys)

    return _make_immutable_config


def make_immutable_sub_spec(e: Dict, keys: List[str]) -> Tuple:
    return tuple([if_list_to_tuple(to_numeric_if_possible(e["config"].get(k, "NOT_EXIST"))) for k in keys])


def make_mutable_sub_spec(e: Dict, keys: List[str]) -> Dict:
    return {k: e["config"][k] for k in keys}


def find_largest_subset(
    exps: List[Dict],
    key: Union[str, List[str]],
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
) -> List[Dict]:
    keys = key if isinstance(key, list) else [key]
    keys.extend(keys_to_ignore)
    partitions = partition_exps(exps, keys)
    largest_partition = max(partitions, key=lambda x: len(x))
    exps = [e for i, e in enumerate(exps) if i in largest_partition]
    return exps


def get_simple_coreset_related_hypers(exp):
    config = exp["config"]
    keys = [k for k in config if "," not in k and "coreset" == k[:7]]
    return [k for k in keys if k not in {"coreset_n_tasks"}]


def sort_by_key(exps: List[Dict], key):
    try:
        return sorted(exps, key=lambda e: to_int_if_possible(e["config"][key]))
    except TypeError as e:
        print(f"can't sort due to TypeError: {e}")
        return exps


def select_exps_differ_by_key(
    exps: List[Dict], key: Union[str, List[str]], add_keys_to_ignore = []
) -> List[Dict]:
    """
    Select the largest subset of exps that differ only by `key`
    """
    to_ignore = list(DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON) + add_keys_to_ignore
    sub_exps = find_largest_subset(exps, key, keys_to_ignore=to_ignore)
    sub_exps = deduplicate(sub_exps)
    sub_exps = sort_by_key(sub_exps, key=key)
    return sub_exps


def distinct_vals(exps: List[Dict], key):
    return {e["config"][key] for e in exps}


def get_slurm_job_info(exp_folder: str):
    prog = re.compile("^slurm_job_(?P<job_id>\d+)_(?P<task_id>\d+)$")
    for file in os.listdir(exp_folder):
        match = prog.match(file)
        if match:
            # example {'job_id': '40970', 'task_id': '211'}
            return match.groupdict()


def move(src, dst):
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"failed to move from {src} to {dst} due to {e}")
        return False


def move_failed_slurm_log(src: str, exp_subdir: str, dry_run: bool = False):
    exp_subdir = Path(exp_subdir)
    successes = []
    for folder in os.listdir(exp_subdir):
        dst_folder_path = exp_subdir / folder
        job_info = get_slurm_job_info(dst_folder_path)
        if job_info:
            slurm_output_file = f"slurm-{job_info['job_id']}_{job_info['task_id']}.out"
            src_file_path = Path(src) / slurm_output_file
            if not os.path.exists(src_file_path):
                continue
            if dry_run:
                print(f"move from {src_file_path} to {dst_folder_path}")
            else:
                success = move(src=src_file_path, dst=dst_folder_path)
                successes.append(success)
    print(f"moved {sum(successes)} files.")


def exps_to_df(
    exps: List[Dict], keys: List[str], val_lambda: Callable, val_col="value"
) -> pd.DataFrame:
    data = defaultdict(list)
    for e in exps:
        for k in keys:
            data[k].append(to_int_if_possible(e["config"][k]))
        data[val_col].append(val_lambda(e))
    df = pd.DataFrame(data)
    if len(df) and keys:
        df = df.sort_values(by=keys)
    return df


def show_dict_structure(struct: Dict, intend: int = 0):
    pf = "\t" * intend
    lprint = lambda *args: print(pf + str(args[0]), *args[1:])
    for key in struct:
        lprint("-" * 10)
        lprint(key, type(struct[key]))
        if isinstance(struct[key], dict):
            show_dict_structure(struct[key], intend + 1)
        elif hasattr(struct[key], "__len__") and not isinstance(struct[key], str):
            lprint("length ", len(struct[key]))
        else:
            lprint(struct[key])


def load_raw_training_log(exp: Dict, only_last_task: bool = False, task_id: int = None):
    path = Path(exp["path"])
    tasks = os.listdir(path / "raw_training_log")
    if only_last_task or task_id is not None:
        task_id = max([int(t) for t in tasks]) if only_last_task else task_id
        return load_chkpt(Path(path) / "raw_training_log" / str(task_id))
    else:
        return {int(t): load_chkpt(Path(path) / "raw_training_log" / t) for t in tasks}


def load_up_to_date_training_df(exp: Dict) -> pd.DataFrame:
    log = load_raw_training_log(exp)
    task_ids = sorted(log.keys())
    return pd.concat([log[t]["training_log_dataframe"] for t in task_ids])


def load_multiple_folders(exp_folders: List[str], *args, **kwargs):
    exps = []
    for folder in exp_folders:
        exps.extend(read_folder(folder, *args, **kwargs))
    return exps


def get_failed_runs(
    configs: List[Dict],
    exps_folder: str = None,
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
) -> List[Dict]:
    if not configs:
        return []
    exps_failed = read_folder(
        exps_folder=exps_folder, only_config=True, only_failed=True,
    )
    proprosed_exps = [{"config": c} for c in configs]
    keys_to_consider = list(
        filter(lambda k: k not in keys_to_ignore, list(configs[0].keys()))
    )
    make_immutable_config = get_make_immutable_config(keys=keys_to_consider)
    configs_to_keep = {make_immutable_config(e) for e in exps_failed}
    configs_failed = [
        configs[i]
        for i, e in enumerate(proprosed_exps)
        if make_immutable_config(e) in configs_to_keep
    ]
    print(f"There are {len(exps_failed)} failed experiments")
    print(f"Discovered {len(configs_failed)} configs that failed in experiments")
    return configs_failed


def remove_done_runs(
    configs: List[Dict],
    exps_folder: Union[str, List[str]] = None,
    job_file: Union[str, List[str]] = None,
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
    only_sucessful: bool = True,
) -> List[Dict]:
    keys_to_ignore = list(keys_to_ignore)
    keys_to_ignore.extend(list(ConfigTemplate.KEYS_TO_IGNORE))
    if not configs:
        return configs
    exps_ran = []
    if exps_folder is not None:
        exps_folders = exps_folder if isinstance(exps_folder, list) else [exps_folder]
        exps_folders = [f for f in exps_folders if os.path.exists(f)]
        exps_ran.extend(
            load_multiple_folders(
                exp_folders=exps_folders,
                only_config=True,
                only_successful=only_sucessful,
            )
        )
    job_files = job_file if isinstance(job_file, list) else [job_file]
    for job_file in job_files:
        if job_file and os.path.exists(job_file):
            exps_ran.extend(
                [{"config": c} for c in read_configs_from_job_file(job_file=job_file)]
            )
    proprosed_exps = [{"config": c} for c in configs]
    keys_to_consider = list(
        filter(lambda k: k not in keys_to_ignore, list(configs[0].keys()))
    )
    make_immutable_config = get_make_immutable_config(keys=keys_to_consider)
    configs_to_remove = {make_immutable_config(e) for e in exps_ran}
    configs_to_keep = [
        configs[i]
        for i, e in enumerate(proprosed_exps)
        if make_immutable_config(e) not in configs_to_remove
    ]
    job_files_str = "\n".join(list(map(str, job_files)))
    print(
        f"Removed {len(configs) - len(configs_to_keep)} runs that have been successful in folder {exps_folder}\n"
        f"and in job files {job_files_str}\n"
        f"Left {len(configs_to_keep)} runs"
    )
    return configs_to_keep


def is_timestamp(string: str, pattern: str = "%Y-%m-%d-%A-%H-%M-%S") -> bool:
    try:
        datetime.strptime(string, pattern)
    except ValueError:
        return False
    else:
        return True


def exp_datetime(exp, pattern: str = "%Y-%m-%d-%A-%H-%M-%S"):
    return datetime.strptime(os.path.basename(exp["path"]), pattern)


def sort_by_datetime(exps) -> None:
    exps.sort(key=exp_datetime)


def is_subdir(path: str) -> bool:
    if not os.path.exists(path):
        return True
    folders = os.listdir(path)
    if not all([is_timestamp(f, "%Y-%m-%d-%A-%H-%M-%S") for f in folders]):
        return False
    try:
        get_root_dir_from_subdir(path)
    except AssertionError:
        return False
    else:
        return True


def remove_done_runs_multiple(
    configs: List[Dict],
    runs_folder: str,
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
    only_sucessful: bool = True,
) -> List[Dict]:
    runs_folder = Path(runs_folder)
    print(
        f"checking folder {runs_folder} for duplicated experiments\n"
        f"It has the following subdir folders {os.listdir(runs_folder)}"
    )
    for subdir_folder in os.listdir(runs_folder):
        assert not is_timestamp(
            subdir_folder
        ), f"The folder {runs_folder} doesn't have subdir, call remove_successful_runs instead"
        configs = remove_done_runs(
            configs=configs,
            exps_folder=runs_folder / subdir_folder,
            keys_to_ignore=keys_to_ignore,
            only_sucessful=only_sucessful,
        )
    return configs


def is_oom(e: Dict) -> bool:
    possible_strings = {
        "Out of memory while trying to",
        "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
        "Out Of Memory",
    }
    log = read_slurm_log(e)
    return any([string in log for string in possible_strings])


def aggregate_keys(exps: List[Dict], keys: List[str]) -> str:
    make_immutable_config = get_make_immutable_config(keys)
    aggregated_key = ", ".join(keys)
    for e in exps:
        e["config"][aggregated_key] = make_immutable_config(e)
    return aggregated_key


def distinct_sub_specs(exps: List[Dict], keys: List[str]) -> List[Dict]:
    immutable_subspecs = set()
    sub_specs = []
    for e in exps:
        immutable = make_immutable_sub_spec(e, keys)
        if immutable not in immutable_subspecs:
            sub_specs.append(make_mutable_sub_spec(e, keys))
            immutable_subspecs.add(immutable)
    sub_specs.sort(key=lambda x: tuple([x[k] for k in keys]))
    return sub_specs


def rank_configs(configs: List, key: Union[str, int], value_order: List = None) -> List:
    if value_order:
        order = {b: i for i, b in enumerate(value_order)}
        key_lambda = lambda x: order[x[key]]
    else:
        key_lambda = lambda x: x[key]
    return sorted(configs, key=key_lambda)


def rank_by_order(values, order):
    order = {b: i for i, b in enumerate(order)}
    return sorted(values, key=lambda x: order[x])


def extract_subspecs(exps: List[Dict], pattern: str) -> List[Dict]:
    if not exps:
        return exps
    re_pattern = re.compile(pattern)
    matched_keys = [k for k in exps[0]["config"].keys() if re_pattern.match(k)]
    return [{k: e["config"][k] for k in matched_keys} for e in exps]


def aggregate_coreset_keys(
    exps: List[Dict], coreset_col: str = "coreset", fields=("heuristic", "offset")
) -> str:
    fields_cols = {
        "heuristic": {
            "kl": f"coreset_kl_heuristic",
            "elbo": f"coreset_elbo_heuristic",
            "entropy": f"coreset_entropy_mode",
        },
        "offset": {
            "kl": f"coreset_kl_offset",
            "elbo": f"coreset_elbo_offset",
            "entropy": f"coreset_entropy_offset",
        },
    }

    def make_immutable_config(e: Dict):
        c = e["config"]
        coreset = c[coreset_col]
        l = [coreset]
        for f in fields:
            try:
                l.append(c[fields_cols[f][coreset]])
            except KeyError:
                l.append("NA")
        return tuple(l)

    aggregated_key = ", ".join(["coreset"] + list(fields))
    for e in exps:
        e["config"][aggregated_key] = make_immutable_config(e)
    return aggregated_key


def translate_coreset_tuples(df, coreset_col, translated="coreset_column"):
    def translate(v):
        if v[0] == "random":
            return "random"
        else:
            return f"{v[0]} {v[1]}"

    df[translated] = df[coreset_col].apply(translate)
    return translated


def show_timestamps(exps: List[Dict]) -> List[str]:
    return sorted([e["path"].split("/")[-1] for e in exps])


def get_exps_to_run(
    job_file: str, exps_folder: str, only_successful: bool = True
) -> List[Dict]:
    exps_done = read_folder(
        exps_folder=exps_folder, only_config=True, only_successful=only_successful
    )
    exps_to_run = [{"config": c} for c in read_configs_from_job_file(job_file=job_file)]
    return subtract(exps_to_run, exps_done)


def get_final_avg_accuracy(exp: Dict, mode: str) -> float:
    if mode == "train_or_valid":
        return get_final_avg_valid_accuracy(exp)
    elif mode == "test":
        return get_final_avg_test_accuracy(exp)
    else:
        raise ValueError(
            f"mode should be either 'train_or_valid' or 'test', got {mode}"
        )


def get_final_avg_test_accuracy(exp: Dict, cached=True) -> float:
    """
    The reason that this function is not using chkpt and training_log_df is because
    that dataframe contains training accuracies, contrary to what the column names
    suggested
    """
    if cached and "final_avg_test_accuracy" in exp:
        return exp["final_avg_test_accuracy"]
    raw_log_last_task = load_raw_training_log(exp, only_last_task=True)
    if "batch_log" in raw_log_last_task:
        accuracies = raw_log_last_task["batch_log"]["accuracies_test"][-1]
    else:
        accuracies = raw_log_last_task["records"][-1]["accuracies_test"]
    acc = np.mean(accuracies)
    exp["final_avg_test_accuracy"] = acc
    return acc


def get_final_log_df_accuracy(exp: Dict, cached=True) -> float:
    if cached and "final_avg_test_accuracy" in exp:
        return exp["final_avg_test_accuracy"]
    load_exp(exp)
    df = exp["chkpt"]["training_log_df"]
    max_task_id = df["task_id"].max()
    df = df.loc[df["task_id"] == max_task_id]
    max_epoch = df["epoch"].max()
    df = df.loc[df["epoch"] == max_epoch]
    acc_cols = [c for c in df.columns if "_acc_" in c]
    vals = df[acc_cols].mean(axis=1)
    assert len(vals) == 1, vals
    return vals.iloc[0]


def get_final_avg_valid_accuracy(exp: Dict) -> float:
    assert not exp["config"]["not_use_val_split"]
    training_log_df = exp["chkpt"]["training_log_df"]
    max_task_id = training_log_df["task_id"].max()
    max_epoch = training_log_df.loc[training_log_df["task_id"] == max_task_id][
        "epoch"
    ].max()
    df = training_log_df
    row_df = df.loc[(df["epoch"] == max_epoch) & (df["task_id"] == max_task_id)]
    avg_accu_serie = calculate_avg_accu(row_df)
    assert len(avg_accu_serie) == 1
    return avg_accu_serie.iloc[0]


def get_max_task_id(exp: Dict) -> int:
    if "chkpt" not in exp:
        load_exp(exp)
    max_task_id = exp["chkpt"]["training_log_df"]["task_id"].max()
    return max_task_id


def get_expected_max_task_id(exp: Dict) -> int:
    """
    1-based index
    """
    c = exp["config"]
    if c["data_training"] == "continual_learning_cifar":
        return 6
    else:
        raise ValueError(f"unimplemented: {c['data_training']}")


def calculate_avg_accu(training_log_df: pd.DataFrame) -> pd.Series:
    accu_cols = find_accuracy_cols_from_log_df(training_log_df.columns)
    return training_log_df[accu_cols].mean(axis=1)


def find_accuracy_cols_from_log_df(df_cols: List[str]) -> List[str]:
    return list(filter(lambda c: "_acc_" in c, df_cols))


def get_task_id_from_accuracy_column(accuracy_column: str) -> int:
    """
    The task id is 1 based
    """
    matched = re.search("task(\d+)", accuracy_column)
    if matched:
        return int(matched.group(1))


def has_catastrophic_forgetting(e: Dict, verbose=False) -> bool:
    log = load_raw_training_log(e, only_last_task=True)
    current_task_id_1_based = log["task_id"] + 1
    df_last_task = log["training_log_dataframe"]
    accu_cols = find_accuracy_cols_from_log_df(df_last_task.columns)
    acc_cols_previous_tasks = [
        c
        for c in accu_cols
        if get_task_id_from_accuracy_column(c) < current_task_id_1_based
    ]
    assert len(acc_cols_previous_tasks) == current_task_id_1_based - 1
    # heuristic: look at the last 20% of rows, see if it is under 10% accuracy
    accuracies_previous_tasks = df_last_task[acc_cols_previous_tasks]
    assert (
        len(accuracies_previous_tasks) >= 20
    ), f"only {len(accuracies_previous_tasks)} records"
    n_rows = int(len(accuracies_previous_tasks) * 0.2)
    accuracy_last_20_percent_rows = accuracies_previous_tasks.iloc[-n_rows:]
    if verbose:
        print(
            f"Accuracies of previous tasks at last 20% epochs: \n"
            f"{accuracy_last_20_percent_rows}"
        )
    return accuracy_last_20_percent_rows.mean().mean() < 0.1


def load_args(exp: Dict) -> argparse.Namespace:
    command_line = read_command_line(exp)
    # remove "python ../fsvi-cl/bin/fsvicl"
    cmd_str = " ".join(command_line.split(" ")[2:])
    return CL_TEMPLATE.parse_args(cmd_str=cmd_str, to_dict=False, ignore_unknown=True)


def load_kwargs(exp: Dict) -> Dict:
    from sfsvi.run import _process_kwargs
    from benchmarking.train_and_evaluate_cl import _process_kwargs_protocol
    kwargs = load_args(exp)
    kwargs = _process_kwargs(kwargs)
    _process_kwargs_protocol(kwargs)
    return kwargs


def load_data(exp: Dict) -> List:
    """return DatasetSplit"""
    load_exp(exp)
    kwargs = exp["chkpt"]["hparams"]
    from sfsvi.general_utils.log import Hyperparameters
    from benchmarking.data_loaders.get_data import prepare_data

    hparams = Hyperparameters(**kwargs)
    (
        load_task,
        meta_data
    ) = prepare_data(
        task=hparams.task,
        use_val_split=hparams.use_val_split,
        n_permuted_tasks=hparams.n_permuted_tasks,
        # n_valid=hparams.n_valid,
        # fix_shuffle=hparams.fix_shuffle,
    )
    data = [load_task(task_id=i) for i in range(meta_data["n_tasks"])]
    return data


def load_example_batch(exp, task_id):
    from benchmarking.data_loaders.get_data import make_iterators

    data = load_data(exp)
    task_data = data[task_id]
    task_iterators = make_iterators(task_data, exp["config"]["batch_size"])
    x, y = next(task_iterators.batch_train)
    return x.numpy(), y.numpy()


def load_train_data_one_task(
    exp: Dict, task_id: int, n_samples: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    data_split = load_data(exp)[task_id]
    x, y = sample_data_from_one_task(
        d=data_split, n_samples=n_samples, train=True, shuffle=False
    )
    return x, y


def get_tasks(exp):
    path = Path(exp["path"])
    tasks = os.listdir(path / "raw_training_log")
    # tasks starts from 0
    return sorted([int(t) for t in tasks])


def load_params_in_epochs(exp, task_id) -> Dict:
    log = load_raw_training_log(exp)[task_id]
    list_params = log["batch_log"].get("params_in_epochs", [])
    n_epochs_per_params = exp["config"]["n_epochs_save_params"]
    n_epochs_per_params = (
        exp["config"]["epochs"]
        if n_epochs_per_params == "not_specified"
        else int(n_epochs_per_params)
    )
    list_params.append(log["params"])
    return {(i + 1) * n_epochs_per_params: p for i, p in enumerate(list_params)}


def load_all_train_data(
    exp: Dict, n_samples: int = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    data_splits = load_data(exp)
    data = []
    for data_split in data_splits:
        x, y = sample_data_from_one_task(
            d=data_split, n_samples=n_samples, train=True, shuffle=False
        )
        data.append((x, y))
    return data


def sample_data(
    data: List, n_samples: int = None, train: bool = True, shuffle: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """data is a list of DataSplit"""
    samples = []
    for d in data:
        samples.append(
            sample_data_from_one_task(
                d, n_samples=n_samples, train=train, shuffle=shuffle
            )
        )
    x_arrays, y_arrays = list(zip(*samples))
    print("check y labels: ", [tuple(sorted(np.unique(a))) for a in y_arrays])
    return list(x_arrays), list(y_arrays)


def sample_data_from_one_task(
    d, n_samples: int = None, train: bool = True, shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """d is DataSplit"""
    if n_samples is None:
        n_samples = d.n_train if train else d.n_test
    if train:
        samples = get_samples(d.train, n_samples, d.n_train, shuffle=shuffle)
    else:
        samples = get_samples(d.test, n_samples, d.n_test, shuffle=shuffle)
    x, y = samples
    return x, y


def to_numpy(_tuple):
    return tuple(map(lambda x: x.numpy(), _tuple))


def get_samples(
    data: tf.data.Dataset,
    n_samples: int,
    total_n,
    buffer_size=1000,
    shuffle: bool = True,
):
    if shuffle:
        data = data.shuffle(buffer_size=buffer_size)
    finite_data = to_numpy(next(iter(data.batch(total_n))))
    inds_sampled = np.random.choice(
        total_n, size=min(n_samples, total_n), replace=False
    )
    sampled_data = tuple([x[inds_sampled] for x in finite_data])
    return sampled_data


def load_model(exp: Dict):
    load_exp(exp)
    kwargs = exp["chkpt"]["hparams"]
    from sfsvi.general_utils.log import Hyperparameters
    from benchmarking.data_loaders.get_data import prepare_data
    from sfsvi.fsvi_utils.utils_cl import initialize_random_keys
    from sfsvi.fsvi_utils.initializer import Initializer

    default_config = CL_TEMPLATE.default_config()
    default_config.update(kwargs)

    hparams = Hyperparameters(**default_config)
    kh = initialize_random_keys(seed=hparams.seed)
    (
        _,
        meta_data
    ) = prepare_data(
        task=hparams.task,
        use_val_split=hparams.use_val_split,
        n_permuted_tasks=hparams.n_permuted_tasks,
        n_valid=hparams.n_valid,
        fix_shuffle=hparams.fix_shuffle,
    )
    n_train = meta_data["n_train"]
    input_shape = meta_data["input_shape"]
    output_dim = meta_data["output_dim"]

    initializer = Initializer(
        hparams=hparams,
        input_shape=input_shape,
        output_dim=output_dim,
        stochastic_linearization=hparams.stochastic_linearization,
        n_marginals=1,
    )
    (model, _, apply_fn, state_init, params_init,) = initializer.initialize_model(
        kh.next_key()
    )
    return model


def get_additional_diff_keys(
    exps: List[Dict],
    keys: List[str],
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
) -> List[str]:
    diff_keys = find_diff_keys(exps)
    additional_keys = set(diff_keys) - set(keys) - set(keys_to_ignore)
    return sorted(additional_keys)


def sort_architectures(architectures: List[str]) -> List[str]:
    def nb_params(architecture: str) -> int:
        nb_units = list(map(int, architecture.split("_")[1:]))
        return int(np.prod(nb_units))

    return sorted(architectures, key=nb_params)


def save_figure(fig, rel_path: str, root=FIG_ROOT, **kwargs):
    full_path = root / rel_path
    os.makedirs(full_path.parent, exist_ok=True)
    default = {"bbox_inches": "tight"}
    default.update(kwargs)
    fig.savefig(full_path, dpi=fig.dpi, **default)


def get_node(exp: Dict) -> str:
    string = read_slurm_log(exp)
    return string.split("\n")[0].split(" ")[-1]


def back_to_cpu(e):
    return "Device: cpu" in read_slurm_log(e)


def standard_error(values):
    return np.std(values) / np.sqrt(len(values))


def nb_seeds(vals):
    return len(vals) - np.isnan(vals).sum()


def report_test_accuracies(
    exps,
    keys=["data_training"],
    keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
    test_func: Callable = get_final_avg_test_accuracy,
    metric_name: str = "test_accuracy",
    original: bool = False,
):
    keys_to_ignore = list(keys_to_ignore) + ["seed"]
    for e in exps:
        e[metric_name] = test_func(e)
    diff_keys = sorted(set(find_diff_keys(exps)).union(set(keys)) - set(keys_to_ignore))
    diff_keys = [key for key in diff_keys if key in exps[0]["config"]]
    df = exps_to_df(
        exps, keys=diff_keys, val_lambda=lambda e: e[metric_name], val_col=metric_name,
    )
    if original:
        return df
    grouped = df.groupby(diff_keys) if diff_keys else df
    se = grouped[metric_name].agg([np.mean, np.std, standard_error, nb_seeds]).round(4)
    if se.ndim == 1:
        se = pd.DataFrame(se).T
    return se


# def get_variations(
#     exps: List[Dict],
#     base_config: Dict,
#     keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
# ) -> List[Dict]:
#     base_exp = {"config": base_config}
#     sub_specs = []
#     keys_to_ignore += ("seed",)
#     for e in exps:
#         keys = sorted(set(find_diff_keys([e, base_exp])) - set(keys_to_ignore))
#         sub_specs.append({k: e["config"][k] for k in keys})
#     sub_specs = deduplicate_dict(sub_specs)
#
#     for s in sub_specs:
#         keys = list(s.keys())
#         for k in keys:
#             if s[k] == base_config[k]:
#                 del s[k]
#     sub_specs = [
#         s for s in sub_specs if find_diff_keys(filter_exps(exps, d=s)) == ["seed"]
#     ]
#     return sub_specs


def get_variations_v2(
    exps: List[Dict], keys_to_ignore=DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON,
) -> List[Dict]:
    keys_to_ignore = list(keys_to_ignore) + ["seed"]
    diff_keys = sorted(set(find_diff_keys(exps)) - set(keys_to_ignore))
    sub_specs = distinct_sub_specs(exps, keys=diff_keys)
    return [s for s in sub_specs if s]


def config_diff(config1, config2, only_conflict=False):
    for k in set(config1.keys()).union(set(config2.keys())):
        if config1.get(k) == config2.get(k):
            continue
        if only_conflict and (k not in config1 or k not in config2):
            continue
        print(f"\t{k}\t{config1.get(k, 'N/A')} ---> {config2.get(k, 'N/A')}")


def deduplicate_dict(ds: List[Dict]):
    new_ds = []
    for d in ds:
        if d not in new_ds:
            new_ds.append(d)
    return new_ds


def actual_success(e):
    return "Test accuracies, task 4" in read_slurm_log(e)


def extract_test_accu(e):
    log = read_slurm_log(e)
    lines = log.split("\n")
    acc_lines = [l for l in lines if "Test accuracies, task " in l]
    last_acc_line = acc_lines[-1]
    values = re.search("(?<=all)(.+)", last_acc_line).group(0)
    values = values.replace("=", "").replace("[", "").replace("]", "")
    values = values.strip().split(" ")
    values = [float(v) for v in values if v]
    return values


def extract_test_accu_log(exp: Dict, task_id: int, slurm: bool = False,
                          next: int = 4) -> Union[None, List[float]]:
    log = read_slurm_log(exp) if slurm else read_log(exp)
    lines = log.split("\n")
    motif = f"Learning task {task_id}"
    line_indices = [i for i, l in enumerate(lines) if motif in l]
    if not line_indices:
        return
    for i in range(line_indices[0] + 1, line_indices[0] + next):
        if i >= len(lines):
            break
        if "Accuracies" in lines[i]:
            line = lines[i]
            val_string = line.split("[")[1].strip("]")
            vals = [float(v) for v in val_string.split(",")]
            return vals
    return


def get_max_task_from_log(exp: Dict, slurm=True, next=40000):
    max_task_id = 0
    for task_id in range(1, 52):
        res = extract_test_accu_log(exp, task_id=task_id, slurm=slurm, next=next)
        if res is None:
            return max_task_id
        else:
            max_task_id = task_id


def get_task_accuracy_fn(task_id):
    def _get_acc(exp):
        vals = extract_test_accu_log(exp, task_id=task_id, slurm=True, next=40000)
        if not vals:
            return np.nan
        assert len(vals) == task_id, f"task_id = {task_id}, len(vals) = {len(vals)}"
        return np.mean(vals)
    return _get_acc


def reconstruct_chkpt(e):
    chkpt_path = os.path.join(e["path"], "chkpt")
    if not os.path.isfile(chkpt_path):
        d = {"test_accuracy": extract_test_accu(e)}
        with open(chkpt_path, "wb") as p:
            pickle.dump(d, p)
    else:
        print("chkpt already exists")


def get_average_test_accuracy_per_task(exp: Dict):
    log = load_raw_training_log(exp)
    accuracies = []
    task_ids = sorted(log.keys())
    for task_id in task_ids:
        l = log[task_id]
        accuracies.append(np.mean(l["batch_log"]["accuracies_test"][-1]))
    return np.array(accuracies)


def infer_alphabet_size(exp: Dict, validate: bool = False) -> List[int]:
    n = exp["config"]["n_omniglot_coreset_chars"]
    load_exp(exp)
    ans = [len(y) // n for y in exp["chkpt"]["coreset"]["y"]]
    if validate:
        assert ans == get_alphabet_sizes(exp)
    return ans


def reproduce_task_order_omniglot(exp: Dict) -> List[int]:
    config = exp["config"]
    nb_alphabets = 50
    omniglot_randomize_task_sequence_seed = config["seed"] if config["omniglot_randomize_task_sequence"] else None
    if omniglot_randomize_task_sequence_seed is not None:
        rng_state = np.random.RandomState(omniglot_randomize_task_sequence_seed)
        task_sequences = rng_state.permutation(nb_alphabets)
        new_aids_to_old_aids = {task_sequences[i]: i for i in range(nb_alphabets)}
        return [new_aids_to_old_aids[i] for i in range(nb_alphabets)]
    else:
        return list(range(nb_alphabets))


def get_alphabet_sizes(exp):
    import tensorflow_datasets as tfds
    kwargs_load = dict(
        name='omniglot',
        data_dir=EXPS_ROOT / "baselines/data/",
        batch_size=-1,
        as_supervised=False,
    )
    ds_train = tfds.as_numpy(tfds.load(split='train', **kwargs_load))
    ds_test = tfds.as_numpy(tfds.load(split='test', **kwargs_load))
    y = np.concatenate((ds_train['label'], ds_test['label']))
    alphabet_ids = np.concatenate((ds_train['alphabet'], ds_test['alphabet']))
    task_order = reproduce_task_order_omniglot(exp)
    sizes = []
    for t in task_order:
        inds_alphabet_id = np.flatnonzero(alphabet_ids == t)
        sizes.append(len(np.unique(y[inds_alphabet_id])))
    return sizes


def get_average_test_accuracy_per_task_vcl(exp: Dict):
    if "chkpt" not in exp:
        exp = load_exp(exp)
    return np.nanmean(exp["chkpt"]["result"], axis=1)


def get_final_average_test_accuracy_fromp(exp: Dict):
    if "chkpt" not in exp:
        exp = load_exp(exp)
    return exp['chkpt']['result']["mean_accuracies"][0]


def get_average_test_accuracy_per_task_frcl(exp: Dict) -> np.ndarray:
    max_task_id = get_max_task_from_log(exp, slurm=False, next=4)
    avg_accs = []
    for i in range(1, max_task_id + 1):
        accs = extract_test_accu_log(exp, task_id=i, slurm=False, next=4)
        assert len(accs) == i, f"task_id (1-indexed) = {i}, accs = {accs}"
        avg_accs.append(np.mean(accs))
    return np.array(avg_accs)


def get_first_task_last_entropy(exp: Dict) -> float:
    return load_raw_training_log(exp, task_id=0)["entropy"][-1][0]


def load_task_accuracy(exp: Dict, task_id: int, accu_task_id: int) -> float:
    """
    task_id are 1-indexed
    """
    if "chkpt" not in exp:
        load_exp(exp)
    acc_col_name = f"test_acc_task{accu_task_id}"
    df = exp["chkpt"]["training_log_df"].query(f"task_id == {task_id}")[
        ["epoch", acc_col_name]
    ]
    max_epoch = df["epoch"].max()
    try:
        return df.loc[df["epoch"] == max_epoch, acc_col_name].iloc[0]
    except IndexError:
        pdb.set_trace()


def backward_transfer(exp: Dict, max_task_id: int) -> float:
    transfers = []
    for i in range(1, max_task_id):
        last_acc = load_task_accuracy(exp, task_id=max_task_id, accu_task_id=i)
        acc = load_task_accuracy(exp, task_id=i, accu_task_id=i)
        transfer = last_acc - acc
        transfers.append(transfer)
    return np.mean(transfers)


def forward_transfer(exp: Dict, ind_exps: List[Dict], keys_to_ignore=[]) -> float:
    max_task_id = get_expected_max_task_id(exp)
    keys_to_ignore = set(
        list(DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON)
        + ["epochs"]
        + get_keys_to_ignore_for_one_task(exp["config"])
        + keys_to_ignore
    )
    base_config = {k: v for k, v in exp["config"].items() if k not in keys_to_ignore}
    transfers = []
    for task_id in range(2, max_task_id + 1):
        # don't foget `only_task_id` is zero-based
        base_config["only_task_id"] = task_id - 1
        ind_exp = filter_exps(ind_exps, d=base_config)
        if len(ind_exp) != 1:
            print("different keys for independent exps are", find_diff_keys(ind_exps))
            keys = find_diff_keys(ind_exps + [{"config": base_config}])
            keys = [k for k in keys if k in base_config]
            for k in keys:
                print(
                    k,
                    "independent exp:",
                    ind_exps[0]["config"][k],
                    "base config:",
                    base_config[k],
                )
            pdb.set_trace()
        ind_exp = ind_exp[0]
        acc_ind = load_independent_accuracy(ind_exp)
        acc = load_task_accuracy(exp, task_id=task_id, accu_task_id=task_id)
        transfer = acc - acc_ind
        transfers.append(transfer)
    return np.mean(transfers)


def load_independent_accuracy(exp: Dict) -> float:
    if "chkpt" not in exp:
        load_exp(exp)
    df = exp["chkpt"]["training_log_df"]
    for c in df.columns:
        if "acc_task" in c and c != "test_acc_task1":
            assert df[c].isnull().sum() == df.shape[0]
    return df.loc[df["epoch"] == df["epoch"].max(), "test_acc_task1"].iloc[0]


def get_keys_to_ignore_for_one_task(config: Dict) -> List[str]:
    l = [
        "n_inducing_inputs_first_task",
        "learning_rate_first_task",
    ]
    l.extend([c for c in config if "coreset" in c])
    return l


def simplify_chkpt(chkpt: Dict) -> Dict:
    keys_drop = ["params", "state", "coreset", "hparams"]
    return {k: v for k, v in chkpt.items() if k not in keys_drop}


def simplify_task_raw_log(task_raw_log: Dict, do_simplify_batch_log: bool = True):
    keys_drop = [
        "params",
        "state",
        "coreset",
        "training_log_dataframe",
        "epoch",
        "accuracy",
        "entropy",
        "elbo",
        "loglik",
        "kl",
        "hparams",
    ]
    d = {k: v for k, v in task_raw_log.items() if k not in keys_drop}
    if do_simplify_batch_log and "batch_log" in d:
        d["batch_log"] = simplify_batch_log(batch_log=d["batch_log"])
    return d


def simplify_batch_log(batch_log: Dict):
    return {
        k: v
        for k, v in batch_log.items()
        if "accurac" in k.lower() or "entrop" in k.lower()
    }


def get_subdir(exp: Dict, full_path: bool = False) -> str:
    if full_path:
        return os.path.dirname(exp["path"].rstrip(os.sep))
    else:
        return exp["path"].strip(os.sep).split(os.sep)[-2]


def first_dict_smaller_than_second_dict(d1: Dict, d2: Dict) -> bool:
    assert isinstance(d1, dict) and isinstance(d2, dict), f"d1 = {d1}, d2 = {d2}"
    for k in d1:
        if k not in d2:
            return False
        if isinstance(d1[k], dict):
            if not first_dict_smaller_than_second_dict(d1[k], d2[k]):
                return False
    return True


def safe_save_chkpt(p, **kwargs):
    """
    Only save if the new dict contains more data.
    """
    if os.path.exists(p):
        original = load_chkpt(p)
        if not first_dict_smaller_than_second_dict(original, kwargs):
            return
    save_chkpt(p, **kwargs)


def simplify_subdir(subdir_path: str, new_subdir: str, **kwargs):
    """Simplify a subdirectory and save it to a new sub directory."""
    exps = read_folder(subdir_path, only_successful=True, only_config=True)
    for exp in exps:
        new_run_folder = simplify_exp(
            exp=exp,
            new_subdir=new_subdir,
            clear=False,
            **kwargs
        )
        print(f"Simplified experiment {exp['path']} and saved to {new_run_folder}")


def simplify_exp(
    exp: Dict,
    new_subdir: str,
    clear: bool = False,
    only_keep_last_raw_log: bool = True,
    do_simplify_batch_log: bool = True,
):
    path = Path(exp["path"])
    new_subdir = Path(new_subdir)
    run_name = os.path.basename(exp["path"].strip("/"))
    new_run_folder = new_subdir / run_name
    if clear and new_run_folder.exists():
        shutil.rmtree(new_run_folder)
    load_exp(exp)
    simplified_chkpt = simplify_chkpt(exp["chkpt"])
    if "source_path" in simplified_chkpt:
        simplified_chkpt["source_path_orig"] = simplified_chkpt["source_path"]
        del simplified_chkpt["source_path"]
    chkpt_path = new_run_folder / "chkpt"
    safe_save_chkpt(p=chkpt_path, source_path=exp["path"], **simplified_chkpt)

    new_raw_log_path = new_run_folder / "raw_training_log"
    task_ids = get_tasks(exp)
    if only_keep_last_raw_log:
        task_ids = [max(task_ids)]
    for task_id in task_ids:
        task_raw_log = simplify_task_raw_log(
            load_raw_training_log(exp, task_id=task_id),
            do_simplify_batch_log=do_simplify_batch_log,
        )
        safe_save_chkpt(new_raw_log_path / f"{task_id}", **task_raw_log)

    slurm_files = [f for f in os.listdir(exp["path"]) if "slurm" in f]
    if path != new_run_folder:
        files = ["command_line.txt", "git_commit.txt"] + slurm_files
        for f in files:
            src = path / f
            dest = new_run_folder / f
            shutil.copyfile(src, dest)
    return new_run_folder


def move_directly(exp: Dict, new_subdir: str, clear: bool = True):
    path = Path(exp["path"])
    new_subdir = Path(new_subdir)
    run_name = os.path.basename(exp["path"].rstrip("/"))
    new_run_folder = new_subdir / run_name
    if clear and new_run_folder.exists():
        shutil.rmtree(new_run_folder)
    shutil.copytree(path, new_run_folder, dirs_exist_ok=True)
    return new_run_folder


def creating_simplified_exps(
    exps: List[Dict],
    new_subdir_fn: Callable,
    clear=True,
    verbose=True,
    simplify_exp_fn=simplify_exp,
):
    subdirs = set()
    for e in tqdm(exps):
        subdir = new_subdir_fn(e)
        new_run_folder = simplify_exp_fn(e, new_subdir=subdir, clear=clear)
        if verbose:
            print(f"Moving {e['path']} to {new_run_folder}")
        subdirs.add(subdir)
    return list(subdirs)


def construct_new_subdir(exp: Dict, root_dir: str) -> str:
    path = exp["path"].rstrip(os.sep)
    path, run_folder_name = os.path.split(path)
    path, subdir = os.path.split(path)
    return os.path.join(root_dir, "runs", subdir)


def get_root_dir_from_subdir(subdir: str) -> str:
    path, subdir_name = os.path.split(str(subdir).rstrip(os.sep))
    root_dir, runs = os.path.split(path)
    assert runs == "runs"
    return root_dir


def get_exp_dir_from_subdir(subdir: str) -> str:
    root_dir = get_root_dir_from_subdir(subdir)
    exp_dir, root_dir_name = os.path.split(root_dir)
    return exp_dir


def get_path_relative_to_exp(subdir: str):
    exp_dir = get_exp_dir_from_subdir(subdir)
    rel_path = os.path.relpath(subdir, exp_dir)
    return rel_path


def upload_subdir(subdir: str, bucket: str = "fsvicl_exp", dry_run: bool = True):
    subdir = str(subdir)
    assert is_subdir(subdir), f"{subdir}"
    from sfsvi.exps.utils.sync_to_gcs import main as sync

    rel_path = get_path_relative_to_exp(subdir)
    args = [
        "--src",
        subdir,
        "--dst",
        f"gs://{bucket}/exps/{rel_path}",
        "--parallel",
        "1",
        "--dry-run",
        "1" if dry_run else "0",
    ]
    sync(args)


def download_data(subdir: str, invalid_cache: bool = False, bucket: str = "fsvicl_exp"):
    assert is_subdir(subdir), f"{subdir}"
    from sfsvi.exps.utils.sync_to_gcs import run_command

    if os.path.exists(subdir) and not invalid_cache:
        print(f"Data already exists in {subdir}")
        return
    os.makedirs(subdir, exist_ok=True)
    rel_path = get_path_relative_to_exp(subdir)
    gs_path = f"gs://{bucket}/exps/{rel_path}"
    command = f"gsutil -m rsync -r {gs_path} {subdir}"
    print(f"Data doesn't exist in local directory {subdir}, downloading from GCS {gs_path}")
    run_command(command.split())


def simplify_and_upload(
    exps: List[Dict],
    path: str = "fsvi/exps/plot_data",
    simplify_exp_fn: Callable = simplify_exp,
    verbose: bool = False,
    clear: bool = True,
    dry_run: bool = False,
) -> List[Dict]:
    new_subdir_fn = lambda e: construct_new_subdir(e, root_dir=PROJECT_ROOT / path)
    subdirs = creating_simplified_exps(
        exps,
        new_subdir_fn=new_subdir_fn,
        clear=clear,
        verbose=verbose,
        simplify_exp_fn=simplify_exp_fn,
    )
    exps = []
    for subdir in subdirs:
        upload_subdir(subdir, dry_run=dry_run)
        exps.extend(read_folder(subdir))
    return exps


def recap_hyper_search(exps: List[Dict], original: bool = False,
                       metric_name: str = "test_accuracy"):
    dfs = {}
    add_keys_to_ignore = ["seed"]
    for key in find_diff_keys(exps):
        if key not in DEFAULT_KEYS_TO_IGNORE_DURING_COMPARISON and key not in add_keys_to_ignore:
            sub_exps = select_exps_differ_by_key(
                exps,
                key=key,
                add_keys_to_ignore=add_keys_to_ignore
            )
            df = report_test_accuracies(sub_exps, keys=[key], original=original, metric_name=metric_name)
            dfs[key] = df

    if original:
        return dfs
    keys = sorted(list(dfs.keys()), key=lambda k: dfs[k]["mean"].max() - dfs[k]["mean"].min(), reverse=True)
    for k in keys:
        print("-" * 100)
        print(dfs[k])


def get_prefix(exp):
    cmd_line_str = read_command_line(exp)
    cmd = cmd_line_str.split(" ")
    prefixes = []
    for i, x in enumerate(cmd):
        if x[:2] == "--":
            break
        else:
            prefixes.append(x)
    return " ".join(prefixes)


def get_command_line(exp, update={}) -> str:
    config = deepcopy(exp["config"])
    config.update(update)
    return " ".join([ENTRY_POINT, CL_TEMPLATE.config_to_str(config)])


def remove_warnings_from_stdout(stdout: str):

    def filter_1(line):
        line = line.lower()
        motifs = [
            "tensorflow",
            "warn",
            "numpy",
            "deprecat",
            "things",
            "bug",
            "device",
            "return",
            "dtype=",
            "instructions",
            "long term",
            "consider",
            "conda",
            "tensor",
            "exp_avg",
        ]
        return any([m in line for m in motifs])
    warnings = [filter_1]
    new_stdout = []
    for line in stdout.split("\n"):
        if not any([warn(line) for warn in warnings]):
            new_stdout.append(line)
    return "\n".join(new_stdout)


def convert_vcl_output(stdout: str):
    """
    ('Epoch:', '0001', 'cost=', '161.090194702')
    """
    lines = stdout.split("\n")
    new_lines = []
    for line in lines:
        if "Epoch:" in line:
            tokens = line.strip("(").strip(")").split(",")
