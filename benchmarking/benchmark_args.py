NOT_SPECIFIED = "not_specified"


def add_benchmark_args(parser):
    """
    Common arguments that are used in all trainers
    """
    parser.add_argument(
        "--logroot", type=str, help="The root result folder that store runs for this type of experiment"
    )

    parser.add_argument(
        "--subdir", type=str, help="The subdirectory in logroot/runs/ corresponding to this run"
    )

    parser.add_argument(
        "--save_alt",
        action="store_true",
        default=False,
        help="Whether to save to alternative logging folder",
    )

    parser.add_argument(
        "--data_training",
        type=str,
        default=NOT_SPECIFIED,
        help="Training and in-distribution dataset used (default: not_specified)\n"
             "Examples: 'continual_learning_pmnist', 'continual_learning_smnist', "
             "'continual_learning_sfashionmnist'",
    )

    parser.add_argument(
        "--not_use_val_split",
        action="store_true",
        default=False,
        help="Whether to split a validation set with the same size as test size from training set",
    )

    parser.add_argument(
        "--n_permuted_tasks",
        type=int,
        default=10,
        help="The number of permuted tasks, this is only used when type of CL task is permuted tasks",
    )

    parser.add_argument(
        "--n_omniglot_tasks",
        type=int,
        default=20,
        help="The number of omniglot tasks, this is only used when type of CL task is omniglot tasks",
    )

    parser.add_argument(
        "--n_valid",
        type=str,
        default="same",
        help="The number of validation samples if not_use_val_split=False. If 'same', then the"
             "same number of test samples is used."
    )

    parser.add_argument(
        "--fix_shuffle",
        action="store_true",
        default=False,
        help="If true, fix the shuffle of dataset"
    )

    parser.add_argument(
        "--n_omniglot_coreset_chars",
        type=int,
        default=2,
        help="The number of coreset points per character for omniglot"
    )

    parser.add_argument(
        "--omniglot_randomize_test_split",
        action="store_true",
        default=False,
        help="If True, randomize test split for omniglot using `seed`"
    )

    parser.add_argument(
        "--omniglot_randomize_task_sequence",
        action="store_true",
        default=False,
        help="If True, randomize task sequence for omniglot using `seed`"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    parser.add_argument(
        "--n_coreset_inputs_per_task",
        type=str,
        default=NOT_SPECIFIED,
        help="Number of coreset points per task. The reason that the type is string is that the "
             "default value depends on task, but None is not accepted as integer type.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size to use for training (default: 100)",
    )

    parser.add_argument(
        "--debug_n_train",
        type=int,
        default=None,
        help="Change the number of training samples to this value if set (for debugging)."
    )

    parser.add_argument(
        "--no_artifact",
        action="store_true",
        default=False,
        help="If True, do not store any artifact (for unit testing)"
    )

    parser.add_argument(
        "--data_ood",
        nargs="+",
        default=[NOT_SPECIFIED],
        help="Out-of-distribution dataset used (default: [not_specified]), this argument has deprecated.",
    )

    parser.add_argument(
        "--use_val_split",
        action="store_true",
        default=True,
        help="Whether to split a validation set with the same size as test size from training set,"
             "this option is deprecated, it is --not_use_val_split that is going to take effect.",
    )
