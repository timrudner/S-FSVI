import os
import sys

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from benchmarking.benchmark_args import add_benchmark_args
from benchmarking.method_cl_fromp import MethodCLFROMP
from benchmarking.train_and_evaluate_cl import ContinualLearningProtocol


def add_fromp_args_v2(parser):
	add_benchmark_args(parser)
	parser.add_argument("--n_tasks", type=int)
	parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
	parser.add_argument("--n_layers", type=int, default=2)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--n_epochs", type=int)
	parser.add_argument(
		"--n_points", type=int
	)  # number of memorable points for each task
	parser.add_argument(
		"--select_method",
		type=str,
		# choices={"lambda_ascend", "lambda_descend", "random_choice", "random_noise"}
	)
	parser.add_argument("--tau", type=float)  # should be scaled with n_points
	parser.add_argument(
		"--smnist_eps", type=float, default=1e-6,
	)
	parser.add_argument(
		"--n_steps",
		type=str,
		default="not_specified"
	)


def fromp_v1_to_v2(config):
	config["data_training"] = f"continual_learning_{config['dataset']}"
	del config["dataset"]
	del config["n_seeds"]
	config["save_alt"] = True
	return config


def main(args, orig_cmd=None):
	kwargs = vars(args)
	protocol = ContinualLearningProtocol(orig_cmd, **kwargs)
	cl_method = MethodCLFROMP(
		input_shape=protocol.input_shape,
		output_dim=protocol.output_dim,
		n_coreset_inputs_per_task_list=protocol.n_coreset_inputs_per_task_list,
		range_dims_per_task=protocol.range_dims_per_task,
		kwargs=kwargs,
	)
	return protocol.train(cl_method)


def run_fromp_v2(args, orig_cmd=None):
	return main(args, orig_cmd=orig_cmd)


if __name__ == "__main__":
	run_fromp_v2(sys.argv[1:])
