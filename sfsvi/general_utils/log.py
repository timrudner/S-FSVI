"""Utilities for setting up log folder and storing experiment artifacts."""
import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from retry import retry


PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
EXPS_ROOT = PROJECT_ROOT / "sfsvi/exps"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")


def set_up_logging(
    log_path: str = None, loglevel: int = logging.DEBUG, stdout: bool = False
) -> logging.Logger:
    """Configures and returns loggers.

    :param log_path: path to save the log file.
    :param loglevel: logging level.
    :param stdout: whether to stream the log to stdout.
    :return:
        Logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    logger.handlers = []

    hs = []

    if stdout:
        hs.append(logging.StreamHandler())
    if log_path is not None:
        hs.append(logging.FileHandler(log_path, mode="w"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for h in hs:
        h.setLevel(loglevel)
        h.setFormatter(formatter)
        logger.addHandler(h)
    # this line is to prevent logger to print to stdout
    logger.propagate = False
    return logger


def path_is_relative(path: str) -> bool:
    return path[0] != "/"


def make_path_absolute(path: str, root=EXPS_ROOT) -> Path:
    if path_is_relative(path):
        return root / path
    else:
        return Path(path)


# The retry logic is helpful for avoiding error on Slurm where multiple jobs
# could start simultaneously
@retry(FileExistsError, tries=100, delay=1, backoff=1.5, jitter=(0.1, 0.5))
def create_logdir(
    rootdir: str = None,
    subdir: str = None,
    cmd: List[str] = None,
) -> Path:
    """Creates a unique folder for storing all artifacts of the experiment run,
    saves git repo status, command line argument to the folder.

    :param rootdir: path to the root folder, which can contain multiple
        subdirs.
    :param subdir: path to the subdir, if specified.
    :param cmd: command line arguments.
    :return:
        unique path to the folder for saving all artifact of this experiment
        run.
    """

    if (rootdir is None) or (rootdir == ""):
        p = Path.cwd()
    else:
        p = make_path_absolute(rootdir)

    timestamp = get_timestamp()

    if subdir is not None:
        p = p / "runs" / subdir / timestamp
    else:
        p = p / "runs" / timestamp

    p.mkdir(parents=True)

    save_repo_status(p)
    save_command_line(p, cmd=cmd)
    save_slurm_status(p)

    return p


def save_repo_status(path: Path) -> None:
    """Saves the commit hash and uncommited changes."""
    with open(path / "git_commit.txt", "w") as f:
        subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)

    with open(path / "workspace_changes.diff", "w") as f:
        subprocess.run(["git", "diff"], stdout=f)


def save_command_line(path: Path, cmd: List[str] = None) -> None:
    """Saves the command line arguments."""
    with open(path / "command_line.txt", "w") as f:
        f.write("python " + " ".join(sys.argv if cmd is None else cmd))


def save_slurm_status(path: Path) -> None:
    """Saves the job information about slurm."""
    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if array_job_id is not None:
        Path(path / f"slurm_job_{array_job_id}_{array_task_id}").touch()
    elif job_id is not None:
        Path(path / f"slurm_job_{job_id}").touch()


def save_kwargs(kwargs: Dict, path: Path) -> None:
    """Saves the processed command line arguments."""
    config_header = kwargs.keys()
    with open(path, "a") as config_file:
        config_writer = csv.DictWriter(config_file, fieldnames=config_header)
        config_writer.writeheader()
        config_writer.writerow(kwargs)
        config_file.close()


class Hyperparameters:
    """Represents hyperparameters."""

    def __init__(self, **kwargs):
        self.from_dict(kwargs)

    def from_argparse(self, args: argparse.Namespace):
        self.from_dict(args.__dict__)

    def from_dict(self, d: Dict):
        for k, v in d.items():
            setattr(self, k, v)

    def from_json(self, j):
        d = json.loads(j)
        self.from_dict(d)

    def as_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dict__}

    def to_json(self, path):
        j = json.dumps(self.as_dict(), indent=4, sort_keys=True)
        path.write_text(j)

    def __contains__(self, k):
        return k in self.__dict__

    def __str__(self):
        s = json.dumps(self.as_dict(), indent=4, separators=(",", ":"))
        return f"input args:\n {s}"


def save_chkpt(p: Path, **kwargs) -> None:
    """Saves data as a checkpoint."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, f"wb") as f:
        pickle.dump(kwargs, f)
