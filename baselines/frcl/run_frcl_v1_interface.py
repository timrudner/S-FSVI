def add_frcl_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_size", type=int)  # 2 hidden layers
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_iterations_train", type=int)
    parser.add_argument("--n_iterations_discr_search", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument(
        "--select_method",
        type=str,
        # choices={"random_choice", "random_noise", "trace_term"},
    )
    parser.add_argument("--use_val_split", action="store_true", default=False)
    parser.add_argument(
        "--n_permuted_tasks",
        type=int,
        default=10,
        help="The number of permuted tasks, this is only used when type of CL task is permuted tasks",
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
        "--n_omniglot_inducing_chars",
        type=int,
        default=2,
        help="the number of inducing inputs per character, FRCL paper reported "
             "results on 1, 2, 3"
    )
    parser.add_argument(
        "--n_omniglot_tasks",
        type=int,
        default=50,
        help="the number of omniglot tasks, must be not greater than 50 (FRCL paper used 50 tasks)"
    )
    parser.add_argument(
        "--randomize_test_split",
        action="store_true",
        default=False,
        help="If True, randomize test split using `seed`"
    )
    parser.add_argument(
        "--randomize_task_sequence",
        action="store_true",
        default=False,
        help="If True, randomize task sequence using `seed`"
    )
    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )
