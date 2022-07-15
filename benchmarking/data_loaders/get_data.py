"""Utilities for loading data."""
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import NamedTuple, Iterator
from typing import Union

import numpy as np
import tensorflow as tf

from benchmarking.data_loaders.mnist_and_cifar import get_mnist_or_cifar
from benchmarking.data_loaders.omniglot import get_omniglot
from benchmarking.data_loaders.toy import get_toy_data

DATA_LOAD = {
    "smnist": get_mnist_or_cifar,
    "smnist_sh": get_mnist_or_cifar,
    "sfashionmnist": get_mnist_or_cifar,
    "pmnist": get_mnist_or_cifar,
    "pfashionmnist": get_mnist_or_cifar,
    "cifar_small": get_mnist_or_cifar,
    "cifar100": get_mnist_or_cifar,
    "cifar": get_mnist_or_cifar,
    "omniglot": get_omniglot,
    "toy": get_toy_data,
    "toy_sh": get_toy_data,
    "toy_reprod": get_toy_data,
}

TWO_TUPLE = Tuple[int, int]
TUPLE_OF_TWO_TUPLES = Tuple[TWO_TUPLE, ...]


class Iterators(NamedTuple):
    """
    batch_train: training data iterator with batch_size
    full_train: training data iterator with only one element which is the entire
        training dataset
    full_valid: if validation exists, validation data iterator with only one
        element which is the entire validation dataset
    full_test: test data with only one element which is the entire training dataset
    """

    batch_train: Iterator
    full_train: Iterator
    full_valid: Optional[Iterator]
    full_test: Iterator


class DatasetSplit(NamedTuple):
    """
    `RawData` split into training, validation and test sets for a particular task

    n_train is the size of train, the same for valid and test
    """

    train: tf.data.Dataset
    valid: tf.data.Dataset
    test: tf.data.Dataset
    n_train: int
    n_valid: int
    n_test: int


def prepare_data(
    task: str,
    use_val_split: bool = True,
    fix_shuffle: bool = False,
    n_permuted_tasks: int = 10,
    n_omniglot_tasks: int = 50,
    n_valid: Union[int, str, None] = "same",
    n_omniglot_coreset_chars: int = 2,
    omniglot_dtype=np.float32,
    omniglot_test_random_state: int = 0,
    omniglot_randomize_task_sequence_seed: int = None,
    data_augmentation_seed=None,
    input_dtype=np.float32,
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Returns various dataset information for a specific continual learning set up.

    :param task: determine the type of task
    :param use_val_split: determines the size of training dataset
    :param fix_shuffle: whether to fix the random seed of shuffle of training set.
    :param n_permuted_tasks: number of permuted tasks
    :param n_omniglot_tasks: number of omniglog tasks to run if the task
        sequence is sequential Omniglot, the maximum is 50.
    :param n_valid: the number of data points for validation set
        if "same", then the number of validation data points is set to be equal
        to the number of test data points; if an integer, it specifies the
        number of validation data points; if None, it means there is no
        validation set.
    :param n_omniglot_coreset_chars: the number of context points per character, FRCL paper
        reported results on 1, 2, 3.
    :param omniglot_dtype: data type of input images for sequential Omniglot.
    :param omniglot_test_random_state: random seed for train-test split.
    :param omniglot_randomize_task_sequence_seed: random seed for random
        permutation of the task sequence for the sequential Omniglot.
    :param data_augmentation_seed: seed for generating augmented data (e.g.
        random croping).
    :param input_dtype: data type of input images.

    :return:
        a function to load task-specific data, it takes in task id and returns
            a DatasetSplit object.
        a dictionary of meta data, it contains the following keys:
            n_tasks: the number of tasks in the task sequence.
            n_train: the number of training data points.
            input_shape: shape of input image including batch dimension,
                batch dimension is 1.
            output_dim: the number of output dimensions.
            n_coreset_inputs_per_task_list: the number of maximum allowed
                coreset points for each task. A coreset is a small set of input
                points that can be stored and used when training on future tasks
                for helping avoid forgetting.
            range_dims_per_task: output heads index range for each task.
                For example, for split MNIST (MH), this variable is
                    `range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
                which means output heads for the first task are the 1st and 2nd
                output dimensions, the output heads for the second task are the
                3rd and 4th dimension, etc.
            max_entropies: a sequence of floats, each value corresponds to the
                maximum possible entropy produced by the prediction.
    """
    dataset = re.sub(r"^continual_learning_", "", task)
    data, meta_data = DATA_LOAD[dataset](
        fix_shuffle=fix_shuffle,
        use_val_split=use_val_split,
        n_permuted_tasks=n_permuted_tasks,
        n_valid=n_valid,
        n_omniglot_tasks=n_omniglot_tasks,
        n_omniglot_coreset_chars=n_omniglot_coreset_chars,
        omniglot_dtype=omniglot_dtype,
        omniglot_test_random_state=omniglot_test_random_state,
        omniglot_randomize_task_sequence_seed=omniglot_randomize_task_sequence_seed,
        data_augmentation_seed=data_augmentation_seed,
        dataset=dataset,
        input_dtype=input_dtype,
    )

    def load_task_partial(
        task_id, return_tfds: bool = True
    ) -> Union[DatasetSplit, Dict]:
        d = data[task_id]
        if return_tfds:
            return DatasetSplit(
                _make_tf_dataset(*d["train"]),
                _make_tf_dataset(*d["valid"]) if "valid" in d else None,
                _make_tf_dataset(*d["test"]),
                len(d["train"][0]),
                len(d["valid"][0]) if "valid" in d else 0,
                len(d["test"][0]),
            )
        else:
            return d

    return load_task_partial, meta_data


def _make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
) -> tf.data.Dataset:
    """
    Create infinitely repeated Tensorflow dataset.
    """
    assert len(x) == len(y)
    return tf.data.Dataset.from_tensor_slices((x, y)).repeat()


def make_iterators(task_data: DatasetSplit, batch_size: int) -> Iterators:
    """
    Turn datasets into iterators (like pytorch dataloaders)

    :param task_data: splitted datasets.
    :param batch_size: batch size of training data.
    :return:
        Iterators of data
    """
    iterator_batch_train = iter(task_data.train.batch(batch_size))
    iterator_full_test = iter(task_data.test.batch(task_data.n_test))
    iterator_full_train = iter(task_data.train.batch(task_data.n_train))
    if task_data.valid is None:
        iterator_full_valid = None
    else:
        iterator_full_valid = iter(task_data.valid.batch(task_data.n_valid))
    return Iterators(
        iterator_batch_train,
        iterator_full_train,
        iterator_full_valid,
        iterator_full_test,
    )


def get_output_dim_fn(range_dims_per_task: TUPLE_OF_TWO_TUPLES) -> Callable[[int], int]:
    """Returns a function that outputs the number of output dimension given
    a task id."""
    alphabet_sizes = [h - l for l, h in range_dims_per_task]

    def get_output_dim(task_id):
        """
        Here `task_id` is 0-indexed
        """
        return sum(alphabet_sizes[: (task_id + 1)])

    return get_output_dim
