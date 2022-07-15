"""Configs for S-FSVI."""
import os
from pathlib import Path
from typing import Dict

from sfsvi.exps.utils.config_template import ConfigTemplate
from sfsvi.fsvi_utils.sfsvi_args import add_sfsvi_args
from sfsvi.fsvi_utils.sfsvi_args_v2 import add_sfsvi_args_v2

CL_TEMPLATE = ConfigTemplate(add_args_fn=add_sfsvi_args)
CL_TEMPLATE_v2 = ConfigTemplate(add_args_fn=add_sfsvi_args_v2)
EXPS_ROOT = Path(os.path.dirname(__file__)).parent
ABLATION_ROOT = EXPS_ROOT / "ablation"
TOY_ROOT = EXPS_ROOT / "toy"
SCRATCH_EXPS_ROOT = Path("/scratch/fsvicl/continual_learning_via_function-space_vi/exps")
SCRATCH_ABLATION_ROOT = SCRATCH_EXPS_ROOT / "ablation"


def get_cl_split_MNIST_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 60 --batch_size 128 
        --data_training continual_learning_smnist 
        --model_type fsvi_mlp
        --architecture fc_100 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_inducing_inputs 40 --inducing_input_type uniform_rand --inducing_inputs_bound 0 1
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
        """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_split_MNIST_single_head_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 80 --batch_size 128 
        --data_training continual_learning_smnist_sh 
        --model_type fsvi_mlp
        --architecture fc_100 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_inducing_inputs 40 --inducing_input_type uniform_rand --inducing_inputs_bound 0 1
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
    """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_permuted_MNIST_base_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 10 --batch_size 128 
        --data_training continual_learning_pmnist 
        --model_type fsvi_mlp
        --architecture fc_100_100 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_inducing_inputs 30 --inducing_input_type uniform_rand --inducing_inputs_bound 0 1
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
        """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_split_FashionMNIST_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 60 --batch_size 128 
        --data_training continual_learning_sfashionmnist 
        --model_type fsvi_mlp
        --architecture fc_200_200_200_200 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_inducing_inputs 40 --inducing_input_type uniform_rand --inducing_inputs_bound 0 1
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
        """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_split_SmallCIFAR_config() -> Dict:
    cmd_str = """
        --data_train continual_learning_cifar_small
        --model_type fsvi_cnn --architecture seven_layers --activation relu 
        --prior_type bnn_induced --epochs 100 
        --learning_rate 1e-3 --batch_size 128
        --prior_mean 0. --prior_cov 1 
        --n_inducing_inputs 20 --inducing_input_type uniform_rand 
        --kl_scale none --inducing_inputs_bound 0 1 --n_samples 5 
        --coreset random  
        --seed 0 --save_alt
        --debug
        """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_toy_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 80 --batch_size 128 
        --data_training continual_learning_toy
        --model_type fsvi_mlp
        --architecture fc_100 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_coreset_inputs_per_task 40
        --n_inducing_inputs 40 --inducing_input_type uniform_rand --inducing_inputs_bound 0 1
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
    """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_toy_single_head_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 80 --batch_size 128 
        --data_training continual_learning_toy_sh
        --model_type fsvi_mlp
        --architecture fc_200_200 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 0.001
        --learning_rate 5e-4
        --n_coreset_inputs_per_task 200
        --n_inducing_inputs 80 --inducing_input_type uniform_rand --inducing_inputs_bound 0 2
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
    """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_cl_toy_reprod_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 250 --batch_size 128 
        --data_training continual_learning_toy_reprod
        --model_type fsvi_mlp
        --architecture fc_20_20 --activation relu 
        --prior_type bnn_induced --prior_mean 0. --prior_cov 10.0
        --learning_rate 5e-4
        --n_coreset_inputs_per_task 20
        --n_inducing_inputs 20 --inducing_input_type uniform_rand --inducing_inputs_bound -4 4
        --kl_scale equal 
        --n_samples 5 --n_samples_eval 5 
        --logging 1 --save_alt 
        --not_use_val_split
        --debug
    """
    return CL_TEMPLATE.parse_args(cmd_str)


def get_relative_path_wrt_exps_root(path: str, root=EXPS_ROOT) -> Path:
    return Path(os.path.relpath(path, root))
