"""Good configs based on hyerparameter search."""
from typing import Dict

from sfsvi.exps.utils.config_template import ConfigTemplate
from sfsvi.fsvi_utils.sfsvi_args import add_sfsvi_args

CL_TEMPLATE = ConfigTemplate(add_args_fn=add_sfsvi_args)


def get_best_cl_split_MNIST_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 60 --batch_size 128 
        --data_training continual_learning_smnist 
        --model_type fsvi_mlp
        --architecture fc_400 --activation relu 
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


def get_best_cl_split_MNIST_single_head_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 80 --batch_size 128 
        --data_training continual_learning_smnist_sh 
        --model_type fsvi_mlp
        --architecture fc_400 --activation relu 
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


def get_best_cl_permuted_MNIST_base_config() -> Dict:
    cmd_str = """
        --seed 0
        --epochs 10 --batch_size 128 
        --data_training continual_learning_pmnist 
        --model_type fsvi_mlp
        --architecture fc_400_400 --activation relu 
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


def get_best_cl_split_FashionMNIST_config() -> Dict:
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


def get_best_cl_split_CIFAR_config() -> Dict:
    cmd_str = """
        --data_training continual_learning_cifar --model_type fsvi_cnn --optimizer adam 
        --architecture six_layers --activation relu --prior_mean 0.0 --prior_cov 0.01 --prior_type bnn_induced 
        --epochs 50 --epochs_first_task 200 
        --batch_size 512 --learning_rate 0.0003 --learning_rate_first_task 5e-4 
        --n_coreset_inputs_per_task 200
        --n_inducing_inputs 50 --n_inducing_inputs_first_task 10 --n_inducing_inputs_second_task 200 
        --inducing_input_type train_pixel_rand_0.5 --kl_scale normalized --n_samples 5
        --inducing_inputs_bound 0.0 0.0
        --n_samples_eval 5 --not_use_val_split --coreset random 
        --final_layer_variational 
        --save_path debug --save_alt
        """
    return CL_TEMPLATE.parse_args(cmd_str)
