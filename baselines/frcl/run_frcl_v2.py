"""
New runner of FRCL that uses `benchmarking` framework.
"""
import sys

import numpy as np

from benchmarking.benchmark_args import add_benchmark_args
from benchmarking.method_cl_frcl import MethodCLFRCL
from benchmarking.train_and_evaluate_cl import ContinualLearningProtocol


def add_frcl_args_v2(parser):
	add_benchmark_args(parser)
	parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
	parser.add_argument("--n_layers", type=int, default=2)
	parser.add_argument("--learning_rate", type=float)
	parser.add_argument("--n_iterations_train", type=int)
	parser.add_argument("--n_iterations_discr_search", type=int)
	parser.add_argument(
		"--select_method",
		type=str,
	)


def frcl_v1_to_v2(config):
	config["data_training"] = f"continual_learning_{config['dataset']}"
	del config["dataset"]
	config["n_omniglot_coreset_chars"] = config["n_omniglot_inducing_chars"]
	del config["n_omniglot_inducing_chars"]
	config["omniglot_randomize_test_split"] = config["randomize_test_split"]
	del config["randomize_test_split"]
	config["omniglot_randomize_task_sequence"] = config["randomize_task_sequence"]
	del config["randomize_task_sequence"]
	del config["n_seeds"]
	config["save_alt"] = True
	return config


def main(args, orig_cmd=None):
	kwargs = vars(args)
	protocol = ContinualLearningProtocol(orig_cmd, input_type=np.float64, **kwargs)
	cl_method = MethodCLFRCL(
		input_shape=protocol.input_shape,
		output_dim=protocol.output_dim,
		n_coreset_inputs_per_task_list=protocol.n_coreset_inputs_per_task_list,
		kwargs=kwargs,
	)
	return protocol.train(cl_method)


def run_frcl_v2(args, orig_cmd=None):
	return main(args, orig_cmd=orig_cmd)


if __name__ == "__main__":
	run_frcl_v2(sys.argv[1:])
