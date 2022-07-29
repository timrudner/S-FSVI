import pdb
from typing import Dict, List

from baselines.vcl.run_vcl import add_vcl_args
from baselines.frcl.run_frcl import add_frcl_args
from baselines.fromp.run_fromp import add_fromp_args
from sfsvi.general_utils.log import EXPS_ROOT, PROJECT_ROOT
from sfsvi.exps.utils.config_template import ConfigTemplate

CL_ROOT = PROJECT_ROOT / "fsvi"
FRCL_TEMPLATE = ConfigTemplate(add_args_fn=add_frcl_args)
BASELINE_ROOT = EXPS_ROOT / "baselines"
FRCL_RUNFILE = PROJECT_ROOT / "baselines/frcl/run_frcl.py"
FRCL_PREFIX = f"python {FRCL_RUNFILE}"

FROMP_TEMPLATE = ConfigTemplate(add_args_fn=add_fromp_args)
FROMP_RUNFILE = PROJECT_ROOT / "baselines/fromp/run_fromp.py"
FROMP_PREFIX = f"python {FROMP_RUNFILE}"

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


def baseline_configs_to_file(configs: List[Dict], subdir: str, folder: str = "jobs", root=BASELINE_ROOT):
    # TODO: this is a quite hacky way
    frcl_configs = [c for c in configs if "frcl" in c["template"]]
    fromp_configs = [c for c in configs if "fromp" in c["template"]]
    vcl_configs = [c for c in configs if "vcl" in c["template"]]
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
    VCL_TEMPLATE.configs_to_file(
        configs=vcl_configs,
        file_path=root / folder / f"{subdir}.sh",
        prefix=VCL_PREFIX,
        mode="a",
    )
