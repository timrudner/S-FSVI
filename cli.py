import argparse
from typing import Dict

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
from jax.lib import xla_bridge

from baselines.frcl.run_frcl_v2 import add_frcl_args_v2
from baselines.frcl.run_frcl_v2 import run_frcl_v2
from baselines.fromp.run_fromp_v2 import add_fromp_args_v2
from baselines.fromp.run_fromp_v2 import run_fromp_v2
from baselines.vcl.run_vcl import run_vcl
from sfsvi.exps.utils.baselines_configs import FRCL_TEMPLATE_v2
from sfsvi.exps.utils.baselines_configs import FROMP_TEMPLATE_v2
from sfsvi.exps.utils.baselines_configs import VCL_TEMPLATE
from sfsvi.exps.utils.configs import CL_TEMPLATE_v2
from sfsvi.fsvi_utils.sfsvi_args import add_sfsvi_args
from sfsvi.fsvi_utils.sfsvi_args_v2 import add_sfsvi_args_v2
from sfsvi.fsvi_utils.sfsvi_args_v2 import fsvi_v1_to_v2
from sfsvi.run import run as cl_run
from sfsvi.run_v2 import run as cl_run_v2

print("Jax is running on", xla_bridge.get_backend().platform)


def define_parser():
    parser = argparse.ArgumentParser(description="Function-Space Variation Inference")
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_sfsvi_args(subparsers.add_parser("cl"))
    add_sfsvi_args_v2(subparsers.add_parser("cl_v2"))
    add_frcl_args_v2(subparsers.add_parser("frcl"))
    add_fromp_args_v2(subparsers.add_parser("fromp"))
    return parser


def parse_args():
    return define_parser().parse_args()


def run_config(config: Dict, runner: str = "fsvi") -> None:
    """Perform training with a certain method and configuration.

    :param config: configuration for the training.
    :param runner: name of the continual learning method.
    """
    if runner == "fsvi":
        config = fsvi_v1_to_v2(config)
        cmd = ["cl_v2"] + CL_TEMPLATE_v2.config_to_str(config).split()
        args = define_parser().parse_args(cmd)
        return cl_run_v2(args, orig_cmd=cmd)
    elif runner == "frcl":
        cmd = ["frcl"] + FRCL_TEMPLATE_v2.config_to_str(config).split()
        args = define_parser().parse_args(cmd)
        return run_frcl_v2(args, orig_cmd=cmd)
    elif runner == "fromp":
        cmd = ["fromp"] + FROMP_TEMPLATE_v2.config_to_str(config).split()
        args = define_parser().parse_args(cmd)
        return run_fromp_v2(args, orig_cmd=cmd)
    elif runner == "vcl":
        args = VCL_TEMPLATE.config_to_str(config).split()
        return run_vcl(args)
    else:
        raise NotImplementedError(runner)


def cli():
    """Command line entry point."""
    args = parse_args()

    if args.command == "cl":
        cl_run(args)
    elif args.command == "cl_v2":
        cl_run_v2(args)
    elif args.command == "frcl":
        run_frcl_v2(args)
    elif args.command == "fromp":
        run_fromp_v2(args)
    else:
        raise NotImplementedError(f"Unknown command {args.command}")


if __name__ == "__main__":
    cli()
