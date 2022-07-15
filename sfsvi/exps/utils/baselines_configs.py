"""Configs for baselines."""
from typing import Dict, List

from baselines.frcl.run_frcl_v2 import add_frcl_args_v2
from baselines.fromp.run_fromp_v2 import add_fromp_args_v2
from baselines.vcl.run_vcl import add_vcl_args
from baselines.frcl.run_frcl_v1_interface import add_frcl_args
from baselines.fromp.run_fromp_v1_interface import add_fromp_args
from sfsvi.general_utils.log import EXPS_ROOT, PROJECT_ROOT
from sfsvi.exps.utils.config_template import ConfigTemplate

CL_ROOT = PROJECT_ROOT / "fsvi"
FRCL_TEMPLATE = ConfigTemplate(add_args_fn=add_frcl_args)
FRCL_TEMPLATE_v2 = ConfigTemplate(add_args_fn=add_frcl_args_v2)
BASELINE_ROOT = EXPS_ROOT / "baselines"
FRCL_RUNFILE = PROJECT_ROOT / "baselines/frcl/run_frcl_v1_interface.py"
FRCL_PREFIX = f"python {FRCL_RUNFILE}"
FRCL_PREFIX_v2 = f"fsvi frcl"

FROMP_TEMPLATE = ConfigTemplate(add_args_fn=add_fromp_args)
FROMP_TEMPLATE_v2 = ConfigTemplate(add_args_fn=add_fromp_args_v2)
FROMP_RUNFILE = PROJECT_ROOT / "baselines/fromp/run_fromp_v1_interface.py"
FROMP_PREFIX = f"python {FROMP_RUNFILE}"
FROMP_PREFIX_v2 = f"fsvi fromp"

VCL_TEMPLATE = ConfigTemplate(add_args_fn=add_vcl_args)
VCL_RUNFILE = PROJECT_ROOT / "baselines/vcl/run_vcl.py"
VCL_PREFIX = f"python {VCL_RUNFILE}"


def get_frcl_split_MNIST_config() -> Dict:
    cmd_str = """
        --dataset smnist     --batch_size 128     
        --hidden_size 100     --learning_rate 1e-3     
        --n_iterations_train 2000   --n_iterations_discr_search 1000    
        --seed 42    --n_seeds 1     
        --select_method random_choice
        --save_alt
        """
    return FRCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_frcl_split_MNIST_random_noise_config() -> Dict:
    cmd_str = """
        --dataset smnist     --batch_size 128     
        --hidden_size 100     --learning_rate 1e-3     
        --n_iterations_train 2000     --n_iterations_discr_search 1000     
        --seed 42     --n_seeds 1     
        --select_method random_noise
        --save_alt
        """
    return FRCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_frcl_omniglot_config() -> Dict:
    cmd_str = """
        --dataset omniglot     
        --n_layers 4 --hidden_size 64
        --batch_size 32
        --learning_rate 1e-3
        --n_iterations_train 2500
        --n_iterations_discr_search 1000
        --seed 42
        --select_method trace_term
        --n_omniglot_inducing_chars 2
        --save_alt --n_seeds 1
        --n_omniglot_tasks 50
        """
    return FRCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_fromp_split_MNIST_config() -> Dict:
    cmd_str = """
        --dataset smnist     --n_tasks 5     
        --batch_size 128     --hidden_size 256     
        --lr 1e-4     --n_epochs 15     --seed 42     
        --n_seeds 1     --n_points 40     
        --select_method lambda_descend     
        --tau 10 --save_alt
        """
    return FROMP_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_fromp_split_MNIST_random_noise_config() -> Dict:
    cmd_str = """
        --dataset smnist     --n_tasks 5     
        --batch_size 128     --hidden_size 256     
        --lr 1e-4     --n_epochs 15     --seed 42     
        --n_seeds 1     --n_points 40     
        --select_method random_noise     
        --tau 10 --save_alt
        """
    return FROMP_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_vcl_permuted_MNIST_with_coreset_config() -> Dict:
    cmd_str = """
        --dataset pmnist
        --batch_size 256    --n_epochs 100
        --hidden_size 256  --n_layers 2
        --seed 42     
        --n_coreset_inputs_per_task 200
        --select_method random_choice     
        """
    return VCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_vcl_split_MNIST_with_coreset_config() -> Dict:
    cmd_str = """
        --dataset smnist
        --n_epochs 120
        --hidden_size 256  --n_layers 2  
        --seed 42     
        --n_coreset_inputs_per_task 40
        --select_method random_choice     
        """
    return VCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def baseline_configs_to_file(
    configs: List[Dict],
    subdir: str,
    folder: str = "jobs",
    root=BASELINE_ROOT,
    runner_version="v1",
):
    # TODO: this is a quite hacky way
    frcl_configs = [c for c in configs if "frcl" in c["template"]]
    fromp_configs = [c for c in configs if "fromp" in c["template"]]
    vcl_configs = [c for c in configs if "vcl" in c["template"]]
    if runner_version == "v1":
        FRCL_TEMPLATE.configs_to_file(
            configs=frcl_configs,
            file_path=root / folder / f"{subdir}.sh",
            prefix=FRCL_PREFIX,
            mode="w",
        )
        FROMP_TEMPLATE.configs_to_file(
            configs=fromp_configs,
            file_path=root / folder / f"{subdir}.sh",
            prefix=FROMP_PREFIX,
            mode="a",
        )
    elif runner_version == "v2":
        FRCL_TEMPLATE_v2.configs_to_file(
            configs=frcl_configs,
            file_path=root / folder / f"{subdir}.sh",
            prefix=FRCL_PREFIX_v2,
            mode="w",
        )
        FROMP_TEMPLATE_v2.configs_to_file(
            configs=fromp_configs,
            file_path=root / folder / f"{subdir}.sh",
            prefix=FROMP_PREFIX_v2,
            mode="a",
        )
    else:
        raise NotImplementedError(runner_version)
    VCL_TEMPLATE.configs_to_file(
        configs=vcl_configs,
        file_path=root / folder / f"{subdir}.sh",
        prefix=VCL_PREFIX,
        mode="a",
    )
