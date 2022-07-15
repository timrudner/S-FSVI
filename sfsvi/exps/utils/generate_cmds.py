"""Utilities for generating bash file with one line for each experiment
run."""
import argparse
import json
import os
import stat
from copy import deepcopy
from itertools import product
from typing import Dict, List, Sequence, Callable, Union, Tuple, Any

ADD_ARGS_FN_TYPE = Callable[[argparse.ArgumentParser], None]
ENTRY_POINT = "fsvi cl"


def print_config(config: Dict):
    print("Config:\n", json.dumps(config, indent=4, separators=(",", ":")), "\n")


def generate_configs(
    base_configs: Union[Dict, Sequence[Dict]], key, values: Sequence
) -> List[Dict]:
    if isinstance(base_configs, Dict):
        base_configs = [base_configs]
    new_configs = []
    for v in values:
        for c in base_configs:
            config = deepcopy(c)
            config[key] = v
            new_configs.append(config)
    return new_configs


def generate_configs_sub_specs(base_config: Union[Dict, List[Dict]], specs: List[Dict]) -> List[Dict]:
    base_configs = base_config if isinstance(base_config, list) else [base_config]
    new_configs = []
    for spec in specs:
        for c in base_configs:
            new_config = deepcopy(c)
            new_config.update(spec)
            new_configs.append(new_config)
    return new_configs


def generate_config_single_value(base_config: Dict, key, value) -> Dict:
    new_config = deepcopy(base_config)
    new_config[key] = value
    return new_config


def generate_configs_cartesian(
    base_config: Union[Dict, List[Dict]], kv_pairs: Sequence[Tuple[str, Sequence]]
) -> List[Dict]:
    base_configs = base_config if isinstance(base_config, List) else [base_config]
    keys, values = zip(*kv_pairs)
    configs = []
    for vs in product(*values):
        for conf in base_configs:
            config = deepcopy(conf)
            for k, v in zip(keys, vs):
                config[k] = v
            configs.append(config)
    return configs


def set_configs_key_value(configs: List[Dict], key, value) -> None:
    for c in configs:
        c[key] = value


def set_configs_kv_pairs(
    configs: List[Dict], kv_pairs: Sequence[Tuple[str, Any]]
) -> None:
    for k, v in kv_pairs:
        set_configs_key_value(configs, key=k, value=v)


def deduplicate(configs: List[Dict], keys_to_ignore: List[str]=[]) -> List[Dict]:
    new_configs_index = []
    simplified_configs = [{k: v for k, v in c.items() if k not in keys_to_ignore} for c in configs]
    for i, c in enumerate(simplified_configs):
        new_simplified_configs = [simplified_configs[i] for i in new_configs_index]
        if c not in new_simplified_configs:
            new_configs_index.append(i)
    return [configs[i] for i in new_configs_index]


def generate_bash_file(file_path: str, cmd_strs: List[str], mode: str = "w"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        for s in cmd_strs:
            f.write(s + "\n")
    make_executable(file_path)


def make_executable(file_path: str):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def remove_kl_oom_configs(configs: List[Dict]) -> List[Dict]:
    def is_removing(config):
        OOM_task = {"continual_learning_smnist_sh", "continual_learning_pmnist"}
        return config["coreset"] == "kl" and config["data_training"] in OOM_task
    return list(filter(lambda c: not is_removing(c), configs))


def remove_invalid_coreset_context_points_configs(configs: List[Dict]) -> List[Dict]:
    def in_valid(config):
        return config["n_coreset_inputs_per_task"] >= config["n_inducing_inputs"]
    return list(filter(lambda c: in_valid(c), configs))
