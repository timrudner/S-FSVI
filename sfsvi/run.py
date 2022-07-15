import os

from benchmarking.benchmark_args import NOT_SPECIFIED
from sfsvi.fsvi_utils.sfsvi_args_v2 import fsvi_v1_to_v2

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['PYTHONHASHSEED']=str(0)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# uncomment the following two lines if there is OOM for MLP
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import argparse
import json
import sys
from typing import Dict

from jax.lib import xla_bridge

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_folder)
from benchmarking.train_and_evaluate_cl import ContinualLearningProtocol
from benchmarking.method_cl_fsvi import MethodCLFSVI
from sfsvi.fsvi_utils.sfsvi_args import add_sfsvi_args

tf_cpu_only = True
if tf_cpu_only:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    print('WARNING: TensorFlow is set to only use CPU.')


def define_parser():
    parser = argparse.ArgumentParser(description="Function Space Variational Inference")
    add_sfsvi_args(parser)
    return parser


def parse_args():
    return define_parser().parse_args()


def _process_kwargs(kwargs: Dict) -> Dict:
    kwargs = fsvi_v1_to_v2(kwargs)
    if kwargs["epochs_first_task"] == NOT_SPECIFIED:
        kwargs["epochs_first_task"] = kwargs["epochs"]

    if kwargs["coreset_elbo_n_samples"] == NOT_SPECIFIED:
        kwargs["coreset_elbo_n_samples"] = kwargs["n_samples"]
    kwargs["coreset_elbo_n_samples"] = int(kwargs["coreset_elbo_n_samples"])

    if kwargs["n_context_points"] != NOT_SPECIFIED:
        kwargs["n_context_points"] = int(kwargs["n_context_points"])

    ### copied from run_base.py:
    if "mlp" in kwargs["model_type"]:
        # This assumes the string starts with 'fc_' followed by the number of hidden units for each layer
        kwargs["architecture_arg"] = kwargs["architecture"]
        kwargs["architecture"] = list(map(int, kwargs["architecture"].split("_")[1:]))

    if kwargs["n_condition"] == 0:
        kwargs["n_condition"] = kwargs["batch_size"]
    if kwargs["td_prior_scale"] == 0.0:
        kwargs["td_prior_scale"] = float(kwargs["context_points"])
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
    kwargs["init_logvar_minval"] = float(kwargs["init_logvar_lin"][0])
    kwargs["init_logvar_maxval"] = float(kwargs["init_logvar_lin"][1])
    kwargs["init_logvar_conv_minval"] = float(kwargs["init_logvar_conv"][0])
    kwargs["init_logvar_conv_maxval"] = float(kwargs["init_logvar_conv"][1])
    kwargs["figsize"] = tuple(kwargs["figsize"])
    return kwargs


def run(args, orig_cmd=None):
    kwargs = vars(args)
    kwargs = _process_kwargs(kwargs)
    print(f"\nDevice: {xla_bridge.get_backend().platform}\n")
    print(
        "Input arguments:\n", json.dumps(kwargs, indent=4, separators=(",", ":")), "\n"
    )
    protocol = ContinualLearningProtocol(orig_cmd, **kwargs)
    cl_method = MethodCLFSVI(
        range_dims_per_task=protocol.range_dims_per_task,
        logger=protocol.logger,
        input_shape=protocol.input_shape,
        n_train=protocol.n_train,
        kwargs=kwargs,
        output_dim=protocol.output_dim,
        n_coreset_inputs_per_task_list=protocol.n_coreset_inputs_per_task_list,
    )
    logdir = protocol.train(cl_method)
    print("\n------------------- DONE -------------------\n")

    return logdir


if __name__ == "__main__":
    run(parse_args())
