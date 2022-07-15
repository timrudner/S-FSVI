"""Command line arguments for S-FSVI runner `sfsvi/run_v2.py."""
from typing import Dict

from benchmarking.benchmark_args import NOT_SPECIFIED
from benchmarking.benchmark_args import add_benchmark_args

V1_TO_V2_RENAME = {
    "inducing_inputs_add_mode": "context_points_add_mode",
    "inducing_input_adjustment": "context_point_adjustment",
    "inducing_input_augmentation": "context_point_augmentation",
    "constant_inducing_points": "constant_context_points",
    "n_inducing_input_adjust_amount": "n_context_point_adjust_amount",
    "inducing_input_type": "context_point_type",
    "inducing_input_ood_data": "context_point_ood_data",
    "inducing_input_ood_data_size": "context_point_ood_data_size",
    "n_inducing_inputs": "n_context_points",
    "n_inducing_inputs_first_task": "n_context_points_first_task",
    "n_inducing_inputs_second_task": "n_context_points_second_task",
    "inducing_inputs_bound": "context_points_bound",
    "inducing_points": "context_points",
}


def fsvi_v1_to_v2(config: Dict) -> Dict:
    """Convert v1 config to v2 config.

    :param config: config of an experiment from the first version.
    :return:
      v2 cofnig of experiment.
    """
    return {V1_TO_V2_RENAME.get(k, k): v for k, v in config.items()}


def add_sfsvi_args_v2(parser):
    """Add command line arguments for S-FSVI runner `sfsvi/run.py`.

    :param parser: parser from `argparse` built-in library.
    """
    add_benchmark_args(parser)
    # all subsequent arguments are only for cl

    parser.add_argument(
        "--architecture",
        type=str,
        default=NOT_SPECIFIED,
        help="Architecture of NN (default: not_specified)",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default=NOT_SPECIFIED,
        help="Activation function used in NN (default: not_specified)",
    )

    parser.add_argument(
        "--prior_mean", type=str, default="0", help="Prior mean function (default: 0)"
    )

    parser.add_argument(
        "--prior_cov", type=str, default="0", help="Prior cov function (default: 0)"
    )

    parser.add_argument(
        "--prior_covs",
        nargs="+",
        default=[0.0],
        type=float,
        help="prior_covs used (default: [0.0])",
    )

    parser.add_argument(
        "--prior_type",
        type=str,
        default=NOT_SPECIFIED,
        help="Type of prior (default: not_specified)",
    )

    parser.add_argument(
        "--start_var_opt",
        type=int,
        default=0,
        help="Epoch at which to start optimizing variance parameters (default: 0)",
    )

    parser.add_argument(
        "--learning_rate_var",
        type=float,
        default=1e-3,
        help="Learning rate for logvar paramters (default: 1e-3)",
    )

    parser.add_argument(
        "--dropout_rate", type=float, default=0.0, help="Dropout rate (default: 0.0)"
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0,
        help="Regularization parameter (default: 0)",
    )

    parser.add_argument(
        "--context_points_add_mode",
        type=int,
        default=0,
        help="The strategy for selecting coreset inputs",
    )

    parser.add_argument(
        "--context_point_adjustment",
        action="store_true",
        default=False,
        help="Whether to reduce the number of context inputs after task 2",
    )

    parser.add_argument(
        "--not_use_coreset",
        action="store_true",
        default=False,
        help="Whether to use a coreset or sample context inputs according to "
        "context_point_type (if True, coreset will not have any effect)",
    )

    parser.add_argument(
        "--context_point_augmentation",
        action="store_true",
        default=False,
        help="Whether to append a set of points sampled according to context_point_type to points sampled from coreset",
    )

    parser.add_argument(
        "--plotting",
        action="store_true",
        default=False,
        help="Whether to plot training log after each epoch",
    )

    parser.add_argument("--logging", type=int, default=1, help="Level of logging")

    parser.add_argument(
        "--coreset",
        choices=["random", "entropy", "kl", "elbo", "random_per_class"],
        default="random",
        help="Methods to select coreset from training data of each task",
    )

    parser.add_argument(
        "--coreset_entropy_mode",
        type=str,
        default="soft_highest",
        help="Exact heuristic (used only if coreset == entropy), examples: hard_highest, mixed_lowest",
    )

    parser.add_argument(
        "--coreset_entropy_offset",
        type=str,
        default="0.0",
        help="Offset to apply to soft entropy-based strategies, (used only if coreset == entropy)",
    )

    parser.add_argument(
        "--coreset_kl_heuristic",
        type=str,
        default="lowest",
        help="the heuristic of KL-based coreset selection. Intuitively, highest makes more sense",
    )

    parser.add_argument(
        "--coreset_kl_offset",
        type=str,
        default="0.0",
        help="Offset to apply to soft kl-based strategies, (used only if coreset == kl)",
    )

    parser.add_argument(
        "--coreset_elbo_heuristic",
        type=str,
        default="lowest",
        help="the heuristic of ELBO-based coreset selection.",
    )

    parser.add_argument(
        "--coreset_elbo_offset",
        type=str,
        default="0.0",
        help="Offset to apply to soft ELBO-based strategies, (used only if coreset == elbo)",
    )

    parser.add_argument(
        "--coreset_elbo_n_samples",
        type=str,
        default=NOT_SPECIFIED,
        help="Number of MC samples used for ELBO method for coreset selection."
        "If not specified, it will be same as the n_samples",
    )

    parser.add_argument(
        "--coreset_n_tasks",
        type=str,
        default=NOT_SPECIFIED,
        help="If specified, coreset points from only a subset of tasks will be used in "
        "the KL computation, with the subset size determined by this argument",
    )

    parser.add_argument(
        "--coreset_entropy_n_mixed",
        type=int,
        default=1,
        help="Ratio of number of points to shortlist for mixed entropy-based strategies, "
        "(used only if coreset == entropy)",
    )

    parser.add_argument(
        "--full_ntk",
        action="store_true",
        default=False,
        help="If true, use full covariance matrix of ntk",
    )

    parser.add_argument(
        "--constant_context_points",
        action="store_true",
        default=False,
        help="If true, keep number of context points constant in all tasks. This option"
        " is only possible for single head and when not_use_coreset is True",
    )

    parser.add_argument(
        "--epochs_first_task",
        type=str,
        default=NOT_SPECIFIED,
        help="The number of epochs for the first task, if not specified, then it will be the same as the "
        "--epochs",
    )

    parser.add_argument(
        "--identity_cov",
        action="store_true",
        default=False,
        help="If True, use identity matrix as covariance",
    )

    parser.add_argument(
        "--n_epochs_save_params",
        type=str,
        default=NOT_SPECIFIED,
        help="Save parameter every n epochs",
    )

    parser.add_argument(
        "--n_augment",
        type=str,
        default=NOT_SPECIFIED,
        help="The number of context inputs based on the training data of current task "
        "for tasks after the first task.",
    )

    parser.add_argument(
        "--augment_mode",
        type=str,
        default="constant",
    )

    parser.add_argument(
        "--learning_rate_first_task",
        type=str,
        default=NOT_SPECIFIED,
        help="Learning rate for the first task.",
    )

    parser.add_argument(
        "--save_first_task",
        action="store_true",
        default=False,
        help="If true, save the params and states of first task",
    )

    parser.add_argument(
        "--first_task_load_exp_path",
        type=str,
        default=NOT_SPECIFIED,
        help="The path to the experiment path to load the params and states of the first task",
    )

    parser.add_argument(
        "--only_task_id",
        type=str,
        default=NOT_SPECIFIED,
        help="If specified, continual learning runner will only train model on this one task.",
    )

    parser.add_argument(
        "--loss_type",
        type=int,
        default=1,
        help="The type of loss for continual learning, this argument has deprecated.",
    )

    parser.add_argument(
        "--only_trainable_head",
        action="store_true",
        default=False,
        help="Define output heads that are only going to be trained for the next task to save "
        "memory, especially suited for Omniglot.",
    )

    parser.add_argument(
        "--n_context_point_adjust_amount",
        type=str,
        default=NOT_SPECIFIED,
        help="If context_point_adjustment is True, this is the number of context inputs we are going to "
        "split evenly when the number of tasks encountered so far is smaller than `coreset_n_tasks`",
    )

    parser.add_argument(
        "--save_all_params",
        action="store_true",
        default=False,
        help="If True, save parameters at the end of each task.",
    )

    parser.add_argument(
        "--n_marginals",
        type=int,
        default=1,
        help="Number of marginal dimensions to evaluate the KL supremum over (default: 1)",
    )

    parser.add_argument(
        "--n_condition",
        type=int,
        default=0,
        help="Number of conditioning points for modified batch normalization (default: 0)",
    )

    parser.add_argument(
        "--context_point_type",
        type=str,
        default=NOT_SPECIFIED,
        help="context input selection method (default: not_specified)",
    )

    parser.add_argument(
        "--context_point_ood_data",
        nargs="+",
        default=[NOT_SPECIFIED],
        help="context input ood data distribution (default: [not_specified])",
    )

    parser.add_argument(
        "--context_point_ood_data_size",
        type=int,
        default=50000,
        help="Size of context input ood dataset (default: 50000)",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default=NOT_SPECIFIED,
        help="Model used (default: not_specified). Example: 'fsvi_mlp', 'mfvi_cnn'",
    )

    parser.add_argument(
        "--kl_scale", type=str, default="1", help="KL scaling factor (default: 1)"
    )

    parser.add_argument(
        "--feature_map_jacobian",
        action="store_true",
        default=False,
        help="Use Jacobian feature map (default: False)",
    )

    parser.add_argument(
        "--feature_map_jacobian_train_only",
        action="store_true",
        default=False,
        help="Do not use Jacobian feature map at evaluation time (default: False)",
    )

    parser.add_argument(
        "--feature_map_type",
        type=str,
        default=NOT_SPECIFIED,
        help="Feature map update type (default: not_specified)",
    )

    parser.add_argument(
        "--td_prior_scale",
        type=float,
        default=0.0,
        help="FS-MAP prior penalty scale (default: 0.0)",
    )

    parser.add_argument(
        "--feature_update",
        type=int,
        default=1,
        help="Frequency of feature map updates (default: 1)",
    )

    parser.add_argument(
        "--full_cov", action="store_true", default=False, help="Use full covariance"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of exp log lik samples (default: 1)",
    )

    parser.add_argument(
        "--n_samples_eval",
        type=int,
        default=5,
        help="Number of MC samples for evaluating the BNN prediction in validation or testing, not for training",
    )

    parser.add_argument(
        "--tau", type=float, default=1, help="Likelihood precision (default: 1)"
    )

    parser.add_argument(
        "--noise_std", type=float, default=1, help="Likelihood variance (default: 1)"
    )

    parser.add_argument(
        "--ind_lim",
        type=str,
        default="ind_-1_1",
        help="context point range (default: ind_-1_1)",
    )

    parser.add_argument(
        "--logging_frequency",
        type=int,
        default=10,
        help="Logging frequency in number of epochs (default: 10)",
    )

    parser.add_argument(
        "--figsize",
        nargs="+",
        default=[10, 4],
        help="Size of figures (default: (10, 4))",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="debug",
        help="Path to save results (default: debug), this argument has deprecated.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save output to file, this argument has deprecated.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name (default: " ")",
        nargs="?",
        const="",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate trained model (default: False)",
    )

    parser.add_argument(
        "--resume_training",
        action="store_true",
        default=False,
        help="Resume training, this argument has deprecated.",
    )

    parser.add_argument(
        "--no_final_layer_bias",
        action="store_true",
        default=False,
        help="No bias term in final layer (default: False)",
    )

    parser.add_argument(
        "--extra_linear_layer",
        action="store_true",
        default=False,
        help="additional linear penultimate layer",
    )

    parser.add_argument(
        "--map_initialization",
        action="store_true",
        default=False,
        help="MAP initialization",
    )

    parser.add_argument(
        "--stochastic_linearization",
        action="store_true",
        default=False,
        help="If True, linearize model around sampled parameters instead of mean parameters",
    )

    parser.add_argument(
        "--grad_flow_jacobian",
        action="store_true",
        default=False,
        help="Gradient flow through Jacobian evaluation (default: False)",
    )

    parser.add_argument(
        "--stochastic_prior_mean",
        type=str,
        default="not_specified",
        help="Stochastic prior mean (default: not_specified)",
    )

    parser.add_argument(
        "--batch_normalization",
        action="store_true",
        default=False,
        help="Batch normalization",
    )

    parser.add_argument(
        "--batch_normalization_mod",
        type=str,
        default="not_specified",
        help="Type of batch normalization (default: not_specified)",
    )

    parser.add_argument(
        "--final_layer_variational",
        action="store_true",
        default=False,
        help="If True, linearize BNN about last layer parameters",
    )

    parser.add_argument(
        "--kl_sup",
        type=str,
        default="not_specified",
        help="Type of KL supremum estimation (default: not_specified)",
    )

    parser.add_argument(
        "--kl_sampled",
        action="store_true",
        default=False,
        help="Use Monte Carlo estimate of KL",
    )

    parser.add_argument(
        "--fixed_inner_layers_variational_var",
        action="store_true",
        default=False,
        help="When `final_layer_variational` is True, setting this to True will create variance parameters "
        "for inner layers which won't be optimised during training.",
    )

    parser.add_argument(
        "--init_logvar",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--init_logvar_lin",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar linear layer initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--init_logvar_conv",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar convolutional layer initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--perturbation_param",
        type=float,
        default=0.01,
        help="Linearization parameter pertubation parameter (default: 0.01)",
    )

    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug model"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=NOT_SPECIFIED,
        help="wanbd project (default: not_specified)",
    )

    parser.add_argument(
        "--n_context_points",
        type=str,
        default=NOT_SPECIFIED,
        help="Number of BNN context points (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--n_context_points_first_task",
        type=str,
        default=NOT_SPECIFIED,
        help="Number of BNN context points on first task (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--n_context_points_second_task",
        type=str,
        default=NOT_SPECIFIED,
        help="Number of BNN context points on second task (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--context_points_bound",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="context point range (default: [0,0])",
    )

    parser.add_argument(
        "--use_generative_model",
        action="store_true",
        default=False,
        help="If true, generative model is used instead of coreset",
    )

    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer used (default: adam)"
    )

    parser.add_argument(
        "--optimizer_var",
        type=str,
        default=NOT_SPECIFIED,
        help="Optimizer used for variance paramters (default: not_specified)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum in SGD",
    )

    parser.add_argument(
        "--momentum_var",
        type=float,
        default=0.0,
        help="Momentum in SGD for variance parameters",
    )

    parser.add_argument(
        "--schedule",
        type=str,
        default=NOT_SPECIFIED,
        help="Learning rate schedule type (default: not_specified)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for each task (default: 100)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--context_points",
        type=int,
        default=0,
        help="Number of BNN context points (default: 0)",
    )
