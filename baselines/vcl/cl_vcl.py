
import tensorflow as tf
import numpy as np

from baselines.vcl.alg.data_generator import CustomGenerator, is_single_head
from baselines.vcl.alg import vcl, coreset
from sfsvi.general_utils.log import save_chkpt, create_logdir, set_up_logging


def main(args, orig_cmd=None):
    print("-" * 100)
    print(f"Available GPUs, {tf.config.list_physical_devices('GPU')}")
    if args.no_artifact:
        logger = None
        logdir = None
    else:
        logdir = create_logdir(args.logroot, args.subdir, cmd=orig_cmd)
        logger = set_up_logging(log_path=logdir / "log")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    task = f"continual_learning_{args.dataset}"
    data_gen = CustomGenerator(
        task=task,
        use_val_split=args.use_val_split,
        n_permuted_tasks=args.n_permuted_tasks,
    )

    batch_size = int(args.batch_size) if args.batch_size != "not_specified" else None

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.reset_default_graph()
    # set seed again to avoid the following error on colab
    # "Random ops require a seed to be set when determinism is enabled.
    # Please set a seed before running the op, e.g. by calling
    # tf.random.set_seed(1)."
    tf.random.set_seed(args.seed)
    vcl_result = vcl.run_vcl(
        hidden_size=[args.hidden_size] * args.n_layers,
        no_epochs=args.n_epochs,
        data_gen=data_gen,
        coreset_method=decode_coreset_method(args.select_method),
        coreset_size=args.n_coreset_inputs_per_task,
        batch_size=batch_size,
        single_head=is_single_head(task),
    )
    print(vcl_result)
    if logger:
        logger.info(vcl_result)

    if not args.no_artifact:
        save_chkpt(
            p=logdir / "chkpt", result=vcl_result, hparams=vars(args),
        )
    return logdir


def decode_coreset_method(select_method):
    if select_method == "random_choice":
        coreset_method = coreset.rand_from_batch
    elif select_method == "k-center":
        coreset_method = coreset.k_center
    else:
        raise NotImplementedError(select_method)
    return coreset_method
