"""Utilities for processing and loading data for task sequences based on
MNIST or CIFAR dataset."""
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import sklearn
import sklearn.model_selection
import tensorflow as tf

from benchmarking.data_loaders.utils import get_model_head_settings

METADATA = {
    "smnist": {
        "n_tasks": 5,
        "n_classes": 10,
        "mode": "split",
        "head": "multi",
        "n_coreset_inputs_per_task": 40,
    },
    "smnist_sh": {
        "n_tasks": 5,
        "n_classes": 10,
        "mode": "split",
        "head": "single",
        "n_coreset_inputs_per_task": 200,
    },
    "sfashionmnist": {
        "n_tasks": 5,
        "n_classes": 10,
        "mode": "split",
        "head": "multi",
        "n_coreset_inputs_per_task": 40,
    },
    "pmnist": {
        "n_tasks": 10,
        "n_classes": 10,
        "mode": "permuted",
        "head": "single",
        "n_coreset_inputs_per_task": 200,
    },
    "pfashionmnist": {
        "n_tasks": 10,
        "n_classes": 10,
        "mode": "permuted",
        "head": "single",
        "n_coreset_inputs_per_task": 200,
    },
    "cifar_small": {
        "n_tasks": 5,
        "n_classes": 10,
        "mode": "split",
        "head": "multi",
        "n_coreset_inputs_per_task": 40,
    },
    "cifar100": {
        "n_tasks": 10,
        "n_classes": 100,
        "mode": "split",
        "head": "multi",
        "n_coreset_inputs_per_task": 40,
    },
    "cifar": {
        "n_tasks": 6,
        "n_classes": 60,
        "mode": "split",
        "head": "multi",
        "n_coreset_inputs_per_task": 40,
    },
}


def get_mnist_or_cifar(
    fix_shuffle: bool,
    use_val_split: bool,
    dataset: str,
    n_valid: Union[int, str, None] = None,
    input_dtype=np.float32,
    **kwargs,
):
    """Returns data and meta data for task sequence based on MNIST and
    CIFAR dataset."""
    del kwargs
    if use_val_split:
        assert (
            n_valid is not None
        ), "If using validation split, `n_valid` needs to be provided"
    meta = METADATA[dataset]
    n_tasks = meta["n_tasks"]
    n_classes = meta["n_classes"]
    mode = meta["mode"]
    head = meta["head"]
    x_train, y_train, x_test, y_test = _load_raw_dataset(dataset)
    # TODO: it makes more sense to do stratified shuffling for each task,
    # TODO: this behaviour is kept to not worry about any changes in data.
    x_train, y_train, x_test, y_test = _preprocess_data(
        x_train,
        y_train,
        x_test,
        y_test,
        fix_shuffle=fix_shuffle,
        dataset=dataset,
        dtype=input_dtype,
    )
    if mode == "split":
        task_classes = np.split(np.arange(n_classes), n_tasks)
        data = _split_by_classes(
            task_classes=task_classes,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            use_val_split=use_val_split,
            n_valid=n_valid,
        )
    else:
        data = _get_permuted_data(
            fix_shuffle=fix_shuffle,
            use_val_split=use_val_split,
            n_tasks=n_tasks,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_valid=n_valid,
        )
    range_dims_per_task, max_entropies = get_model_head_settings(
        n_tasks=n_tasks,
        n_classes=n_classes,
        head=head,
    )
    meta_data = {
        "n_tasks": n_tasks,
        "n_train": _get_total_train_samples(data),
        "input_shape": [1] + list(x_train.shape[1:]),
        "output_dim": n_classes,
        "n_coreset_inputs_per_task_list": n_tasks
        * (meta["n_coreset_inputs_per_task"],),
        "range_dims_per_task": range_dims_per_task,
        "max_entropies": max_entropies,
    }
    return data, meta_data


def select_task_examples(
    x: np.ndarray,
    y: np.ndarray,
    task_classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select data points whose labels are in `task_classes`.

    :param x: input.
    :param y: labels.
    :param task_classes: list of labels to include.
    :return:
        input of selected data points.
        label of selected data points.
    """
    inds_task = np.flatnonzero([y_i in task_classes for y_i in y])
    return x[inds_task], y[inds_task]


def _load_raw_dataset(
    task: str,
):
    """Load raw dataset.

    The default path for saving data is at `~/.keras/datasets`.

    :param task: name of the task sequence.
    :return:
    """
    if task in {"smnist", "smnist_sh", "pmnist"}:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif task in {"sfashionmnist", "pfashionmnist"}:
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.fashion_mnist.load_data()
    elif task == "cifar_small":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif task == "cifar100":
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.cifar100.load_data()
    elif task == "cifar":
        n_cifar100_classes = 50
        (x_train_10, y_train_10), (
            x_test_10,
            y_test_10,
        ) = tf.keras.datasets.cifar10.load_data()
        (x_train_100, y_train_100), (
            x_test_100,
            y_test_100,
        ) = tf.keras.datasets.cifar100.load_data()
        x_train_100, y_train_100 = select_task_examples(
            x_train_100, y_train_100, np.arange(n_cifar100_classes)
        )
        x_test_100, y_test_100 = select_task_examples(
            x_test_100, y_test_100, np.arange(n_cifar100_classes)
        )
        y_train_100 += 10
        y_test_100 += 10
        x_train = np.concatenate((x_train_10, x_train_100))
        y_train = np.concatenate((y_train_10, y_train_100))
        x_test = np.concatenate((x_test_10, x_test_100))
        y_test = np.concatenate((y_test_10, y_test_100))
    else:
        raise NotImplementedError(task)
    return x_train, y_train, x_test, y_test


def _preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    fix_shuffle: bool,
    dataset: str,
    dtype=np.float32,
):
    """Apply preprocessing to data."""
    x_train, y_train = _reorder_data(
        x=x_train,
        y=y_train,
        n_valid=len(y_test),
        fix_shuffle=fix_shuffle,
    )
    x_train = tf.image.convert_image_dtype(x_train, dtype).numpy()
    x_test = tf.image.convert_image_dtype(x_test, dtype).numpy()
    if dataset in {"smnist", "smnist_sh", "pmnist", "sfashionmnist", "pfashionmnist"}:
        x_train = _flatten(x_train)
        x_test = _flatten(x_test)
    y_train = _squeeze(y_train)
    y_test = _squeeze(y_test)
    return x_train, y_train, x_test, y_test


def _flatten(x: np.ndarray) -> np.ndarray:
    """Flatten non-batch dimensions."""
    input_dim = np.prod(x.shape[1:])
    x = np.reshape(x, (-1, input_dim))
    return x


def _squeeze(y: np.ndarray) -> np.ndarray:
    """Squeeze labels and sanity check that the label of each data point is
    a scalar."""
    if y.ndim == 2:
        y = np.squeeze(y, axis=1)
    assert y.ndim == 1, f"y.shape = {y.shape}"
    return y


def _reorder_data(
    x: np.ndarray,
    y: np.ndarray,
    n_valid: int,
    fix_shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder data using stratified split to prepare for train-validation
    split."""
    x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        x,
        y,
        test_size=n_valid,
        random_state=(0 if fix_shuffle else None),
        shuffle=True,
        stratify=y,
    )
    x = np.concatenate((x_train, x_valid))
    y = np.concatenate((y_train, y_valid))
    return x, y


def _get_total_train_samples(data):
    return sum([d["train"][1].shape[0] for d in data])


def _split_by_classes(
    task_classes: List[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    use_val_split: bool,
    n_valid: Union[int, str, None],
) -> List[Dict[str, List[np.ndarray]]]:
    """Generate task-specific data for split task sequences and perform optional
    validation split."""
    data = []
    for task_id in range(len(task_classes)):
        current_task_classes = task_classes[task_id]
        x_train_t, y_train_t = select_task_examples(
            x_train, y_train, current_task_classes
        )
        x_test_t, y_test_t = select_task_examples(x_test, y_test, current_task_classes)
        if use_val_split:
            n_valid_t = (
                len(x_test_t)
                if n_valid == "same"
                else np.min([int(n_valid), len(x_test_t)])
            )
            data_t = {
                "train": [x_train_t[:-n_valid_t], y_train_t[:-n_valid_t]],
                "valid": [x_train_t[-n_valid_t:], y_train_t[-n_valid_t:]],
                "test": [x_test_t, y_test_t],
            }
        else:
            data_t = {
                "train": [x_train_t, y_train_t],
                "test": [x_test_t, y_test_t],
            }
        data.append(data_t)
    return data


def _get_permuted_data(
    fix_shuffle: bool,
    use_val_split: bool,
    n_tasks: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_valid: int,
) -> List[Dict[str, List[np.ndarray]]]:
    """Generate task-specific data for permuted task sequences and perform
    optional validation split."""
    if fix_shuffle:
        rng = np.random.default_rng(0)
    else:
        rng = np.random
    task_permutations = []
    input_dim = np.prod(x_train.shape[1:])
    for _ in range(n_tasks):
        permutation = rng.permutation(input_dim)
        task_permutations.append(permutation)

    data = []
    for task_id in range(n_tasks):
        x_train_t = x_train[:, task_permutations[task_id]]
        x_test_t = x_test[:, task_permutations[task_id]]
        if use_val_split:
            n_valid_t = (
                len(x_test_t)
                if n_valid == "same"
                else np.min([int(n_valid), len(x_test_t)])
            )
            data_t = {
                "train": [x_train_t[:-n_valid_t], y_train[:-n_valid_t]],
                "valid": [x_train_t[-n_valid_t:], y_train[-n_valid_t:]],
                "test": [x_test_t, y_test],
            }
        else:
            data_t = {
                "train": [x_train_t, y_train],
                "test": [x_test_t, y_test],
            }
        data.append(data_t)
    return data
