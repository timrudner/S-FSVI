import os

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['PYTHONHASHSEED']=str(0)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# uncomment the following two lines if there is OOM for MLP
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import argparse
import json
import sys
from typing import Dict
from copy import copy

from jax.lib import xla_bridge

root_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_folder)
sys.path.insert(0, os.path.join(root_folder, "function_space_variational_inference"))
from benchmarking.train_and_evaluate_cl import ContinualLearningProtocol
from benchmarking.method_cl_fsvi import MethodCLFSVI
from sfsvi.general_utils.log import save_kwargs
from sfsvi.fsvi_utils.args_cl import add_cl_args, NOT_SPECIFIED

tf_cpu_only = True  # TODO: check how this affects determinism -- keep set to False
if tf_cpu_only:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    print('WARNING: TensorFlow is set to only use CPU.')

# FOR DEBUGGING
# debug = True
debug = False
if debug:
    from jax import config
    config.update('jax_disable_jit', True)
    # config.update('jax_platform_name', 'cpu')
    # config.update("jax_enable_x64", True)


def train_fsvi(orig_cmd, **kwargs):
    benchmark = ContinualLearningProtocol(orig_cmd, **kwargs)
    cl_method = MethodCLFSVI(
        range_dims_per_task=benchmark.range_dims_per_task,
        logger=benchmark.logger,
        input_shape=benchmark.input_shape,
        n_train=benchmark.n_train,
        method_kwargs=kwargs,
        output_dim=benchmark.output_dim,
        n_coreset_inputs_per_task_list=benchmark.n_coreset_inputs_per_task_list,
    )
    return benchmark.train(cl_method)


def main(kwargs, orig_cmd=None):
    # Select trainer
    # trainer = ContinualLearning().run
    # trainer = ContinualLearningProtocol().train

    # PASS VARIABLES TO TRAINER
    logdir = train_fsvi(orig_cmd=orig_cmd, **kwargs)

    if kwargs["save"] and not kwargs["resume_training"]:
        save_path = kwargs["save_path"]
        print(
            f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n"
        )
        os.rename(save_path, f"{save_path}__complete")

    print("\n------------------- DONE -------------------\n")
    return logdir


def define_parser():
    parser = argparse.ArgumentParser(description="Function Space Variational Inference")
    add_cl_args(parser)
    return parser


def parse_args():
    return define_parser().parse_args()


def process_args(args: argparse.Namespace) -> Dict:
    """
    This is the only place where it is allowed to modify kwargs

    This function should not have side-effect.

    @param args: input arguments
    @return:
    """
    kwargs = vars(args)
    if args.data_ood == [NOT_SPECIFIED]:
        task = args.data_training
    else:
        task = f"{args.data_training}_{args.data_ood}"
    kwargs["task"] = task

    save_path = args.save_path.rstrip()
    if args.save:
        save_path = (
            f"results/{save_path}/{task}/model_{args.model_type}__architecture_{args.architecture}__priormean_{args.prior_mean}__"
            f"priorcov_{args.prior_cov}__optimizer_{args.optimizer}__lr_{args.learning_rate}__bs_{args.batch_size}__"
            f"indpoints_{args.n_inducing_inputs}__indtype_{args.inducing_input_type}__klscale_{args.kl_scale}__nsamples_{args.n_samples}__"
            f"tau_{args.tau}__indlim_{args.inducing_inputs_bound[0]}_{args.inducing_inputs_bound[1]}__reg_{args.regularization}__seed_{args.seed}__"
        )
        i = 1
        while os.path.exists(f"{save_path}{i}") or os.path.exists(
            f"{save_path}{i}__complete"
        ):
            i += 1
        save_path = f"{save_path}{i}"
    kwargs["save_path"] = save_path

    kwargs["figsize"] = tuple(kwargs["figsize"])
    if kwargs["not_use_val_split"]:
        kwargs["use_val_split"] = False

    if kwargs["coreset_elbo_n_samples"] == NOT_SPECIFIED:
        kwargs["coreset_elbo_n_samples"] = kwargs["n_samples"]
    kwargs["coreset_elbo_n_samples"] = int(kwargs["coreset_elbo_n_samples"])

    if kwargs["epochs_first_task"] == NOT_SPECIFIED:
        kwargs["epochs_first_task"] = kwargs["epochs"]

    if kwargs["n_inducing_inputs"] != NOT_SPECIFIED:
        kwargs["n_inducing_inputs"] = int(kwargs["n_inducing_inputs"])

    ### copied from run_base.py:
    if "mlp" in kwargs["model_type"]:
        # This assumes the string starts with 'fc_' followed by the number of hidden units for each layer
        kwargs["architecture_arg"] = kwargs["architecture"]
        kwargs["architecture"] = list(map(int, kwargs["architecture"].split("_")[1:]))

    if kwargs["n_condition"] == 0:
        kwargs["n_condition"] = kwargs["batch_size"]
    if kwargs["td_prior_scale"] == 0.0:
        kwargs["td_prior_scale"] = float(kwargs["inducing_points"])
    if kwargs["prior_covs"][0] != 0.0:
        prior_cov = []
        for i in range(len(kwargs["prior_covs"])):
            prior_cov.append(kwargs["prior_covs"][i])
        kwargs["prior_cov"] = prior_cov

    if kwargs["feature_map_type"] == "learned_nograd":
        kwargs["grad_flow_jacobian"] = False
    elif kwargs["feature_map_type"] == "learned_grad":
        kwargs["grad_flow_jacobian"] = True

    kwargs["init_logvar_minval"] = float(kwargs["init_logvar"][0])
    kwargs["init_logvar_maxval"] = float(kwargs["init_logvar"][1])
    kwargs["init_logvar_lin_minval"] = float(kwargs["init_logvar_lin"][0])
    kwargs["init_logvar_lin_maxval"] = float(kwargs["init_logvar_lin"][1])
    kwargs["init_logvar_conv_minval"] = float(kwargs["init_logvar_conv"][0])
    kwargs["init_logvar_conv_maxval"] = float(kwargs["init_logvar_conv"][1])
    kwargs["figsize"] = tuple(kwargs["figsize"])

    return kwargs


def run(args, orig_cmd=None):
    kwargs = process_args(args)
    # all subsequent code should not modify kwargs

    if kwargs["save"]:
        # Automatically makes parent directories
        os.makedirs(f"{kwargs['save_path']}/figures", exist_ok=True)
        save_kwargs(kwargs=kwargs, path=f"{kwargs['save_path']}/config.csv")

    if kwargs["save"] and not kwargs["debug"]:
        stdout_file = f"{kwargs['save_path']}/stdout.txt"
        print(f"stdout now is saved to {stdout_file}")
        orig_stdout = sys.stdout
        stdout_file = open(stdout_file, "w")
        sys.stdout = stdout_file

    print(f"\nDevice: {xla_bridge.get_backend().platform}\n")

    print(
        "Input arguments:\n", json.dumps(kwargs, indent=4, separators=(",", ":")), "\n"
    )

    kwargs['kwargs'] = copy(kwargs)

    logdir = main(kwargs, orig_cmd=orig_cmd)

    if kwargs["save"] and not kwargs["debug"]:
        sys.stdout = orig_stdout
        stdout_file.close()
    return logdir


if __name__ == "__main__":
    run(parse_args())
