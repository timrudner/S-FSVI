import argparse
import subprocess
import sys
from typing import List


def run_command(cmds: List[str]):
    p = subprocess.Popen(
        " ".join(cmds), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", shell=True
    )
    output, error = p.communicate()
    sep = "*" * 100 + "\n"
    if p.returncode != 0:
        raise Exception(
            f"failed, return code = {p.returncode}\n {sep}output: {output} \n {sep}error: {error}\n"
        )
    return output, error


def format_cmd_run_result(output, error):
    seps = "-" * 100
    return f"{seps}stdout:\n{output}\n\n{seps}stderr:\n{error}\n\n"


def parse_args(args):
    parser = argparse.ArgumentParser(description="Moving data")
    parser.add_argument("-s", "--src", type=str, required=True, help="The source path")
    parser.add_argument(
        "-d",
        "--dst",
        type=str,
        required=True,
        help="The destination path of GCS. "
        "Example: gs://<bucket_name>/<path within bucket>",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="1 means perform rsync in parallel, 0 means not in parallel",
    )

    parser.add_argument(
        "--dry-run",
        type=int,
        default=1,
        help="1 means this is a dry-run, 0 means not dry-run"
        "In order to start copying, turn this option off",
    )

    parser.add_argument(
        "--only",
        nargs="*",
        type=str,
        default=[],
        help="Only files whose paths contain these strings are synchronised",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        type=str,
        default=[],
        help="arguments passed to -x option of gsutil rsync, regular expression is allowed, see "
        "the documentation https://cloud.google.com/storage/docs/gsutil/commands/rsync#options",
    )

    parsed = parser.parse_args(args)
    print_args(parsed)
    return parsed


def print_args(parsed: argparse.Namespace):
    for k, v in vars(parsed).items():
        print(f"{k}: {v}")


def make_re_filter(only_included_list: List[str], excluded_list: List[str]):
    filters = [f"^((?!{name}).)*$" for name in only_included_list]
    filters.extend(excluded_list)
    filter_string = "|".join(filters)
    return f"'{filter_string}'"


def construct_command(args: argparse.Namespace) -> List[str]:
    cmd = ["gsutil", "rsync", "-r"]
    if args.parallel:
        cmd.insert(1, "-m")
    if args.dry_run:
        cmd.append("-n")
    if args.only or args.exclude:
        cmd.extend(["-x", make_re_filter(args.only, args.exclude)])
    cmd.extend([args.src, args.dst])
    return cmd


def main(args):
    args = parse_args(args)
    command = construct_command(args)
    print(f"constructed the following command:\n {' '.join(command)}")
    output, error = run_command(command)
    print(format_cmd_run_result(output, error))


if __name__ == "__main__":
    main(sys.argv[1:])
