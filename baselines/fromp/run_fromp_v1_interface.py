def add_fromp_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_tasks", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_seeds", type=int)
    parser.add_argument(
        "--n_points", type=int
    )  # number of memorable points for each task
    parser.add_argument(
        "--select_method",
        type=str,
        # choices={"lambda_ascend", "lambda_descend", "random_choice", "random_noise"}
    )
    parser.add_argument("--tau", type=float)  # should be scaled with n_points
    parser.add_argument("--use_val_split", action="store_true", default=False)
    parser.add_argument(
        "--n_permuted_tasks",
        type=int,
        default=10,
        help="The number of permuted tasks, this is only used when type of CL task is permuted tasks",
    )
    parser.add_argument(
        "--smnist_eps", type=float, default=1e-6,
    )
    parser.add_argument(
        "--logroot",
        type=str,
        help="The root result folder that store runs for this type of experiment",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        help="The subdirectory in logroot/runs/ corresponding to this run",
    )
    parser.add_argument(
        "--save_alt",
        action="store_true",
        default=False,
        help="Whether to save to alternative logging folder",
    )
    parser.add_argument(
        "--n_coreset_inputs_per_task",
        type=str,
        default="not_specified",
        help="Number of coreset points per task. The reason that the type is string is that the "
             "default value depends on task, but None is not accepted as integer type.",
    )
    parser.add_argument(
        "--n_steps",
        type=str,
        default="not_specified"
    )
    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )
