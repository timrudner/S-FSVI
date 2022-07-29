import argparse
from typing import Dict

from baselines.frcl.run_frcl import run_frcl
from baselines.fromp.run_fromp import run_fromp
from baselines.vcl.run_vcl import run_vcl
from sfsvi.exps.utils.baselines_configs import FRCL_TEMPLATE
from sfsvi.exps.utils.baselines_configs import FROMP_TEMPLATE
from sfsvi.exps.utils.baselines_configs import VCL_TEMPLATE
from sfsvi.exps.utils.configs import CL_TEMPLATE

tf_cpu_only = True  # TODO: check how this affects determinism -- keep set to False
if tf_cpu_only:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    print('WARNING: TensorFlow is set to only use CPU.')
from jax.lib import xla_bridge
print("Jax is running on", xla_bridge.get_backend().platform)

from sfsvi.fsvi_utils.args_cl import add_cl_args
from sfsvi.run import run as cl_run


def define_parser():
    parser = argparse.ArgumentParser(description="Function-Space Variation Inference")
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_cl_args(subparsers.add_parser("cl"))
    return parser


def parse_args():
    return define_parser().parse_args()


def run_config(config: Dict, runner: str = "fsvi"):
    if runner == "fsvi":
        cmd = ["cl"] + CL_TEMPLATE.config_to_str(config).split()
        args = define_parser().parse_args(cmd)
        return cl_run(args, orig_cmd=cmd)
    elif runner == "frcl":
        args = FRCL_TEMPLATE.config_to_str(config).split()
        return run_frcl(args)
    elif runner == "fromp":
        args = FROMP_TEMPLATE.config_to_str(config).split()
        return run_fromp(args)
    elif runner == "vcl":
        args = VCL_TEMPLATE.config_to_str(config).split()
        return run_vcl(args)
    else:
        raise NotImplementedError(runner)


def cli():
    args = parse_args()

    if args.command == "cl":
        cl_run(args)
    else:
        raise NotImplementedError(f"Unknown command {args.command}")


if __name__ == "__main__":
    cli()
