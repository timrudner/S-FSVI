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
    n_omniglot_tasks: int = 20,  # Standard Omniglot: 50
    n_valid: str = "same",
    n_omniglot_coreset_chars: int = 2,
    omniglot_dtype=np.float32,
    omniglot_test_random_state: int = 0,
    omniglot_randomize_task_sequence_seed: int = None,
    data_augmentation_seed=None,
    input_dtype=np.float32,
) -> Tuple[
    Callable,
    Dict[str, Any],
]:
    """
    Returns various dataset information for a specific continual learning set up.

    @param task: determine the type of task
    @param fix_shuffle: whether to fix the random seed of shuffle of training set.
    @param use_val_split: determines the size of training dataset
    @param n_permuted_tasks: number of permuted tasks
    @param n_omniglot_coreset_chars: the number of context points per character, FRCL paper
        reported results on 1, 2, 3
    @return:
        load_task_partial: the function that takes in task id and returns a DatasetSplit object
        n_tasks: the number of tasks
        n_train: the number of training samples
        n_coreset_inputs_per_task: a tuple of integers, the number of context points for each task
        input_shape: the shape of input data with batch dimension to be 1
        output_dim: output dimension of task
        range_dims_per_task: a sequence of 2-tuples, each 2-tuple (start, end) corresponds to one task,
            the output head is from `start` till `end` (exclusive)
        max_entropies: a sequence of floats, each value corresponds to the maximum possible entropy
            produced by the prediction
    """
    # remove `continual_learning_` prefix from task to get the dataset to load
    dataset = re.sub(r"^continual_learning_", "", task)
    # TODO: n_train -- remove hardcoded values; check values (can affect coreset selection)
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

    def load_task_partial(task_id, return_tfds: bool = True) -> Union[DatasetSplit, Dict]:
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
    Create infinitely repeated Tensorflow dataset, the first dimension of x and y must
    correspond to different samples
    """
    assert len(x) == len(y)
    return tf.data.Dataset.from_tensor_slices((x, y)).repeat()


def make_iterators(task_data: DatasetSplit, batch_size: int) -> Iterators:
    """
    Turn datasets into iterators (like pytorch dataloaders)

    @param task_data: contains train, valid, and test datasets
    @param batch_size: batch size of training data
    @return:
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
        iterator_batch_train, iterator_full_train, iterator_full_valid, iterator_full_test
    )


def get_output_dim_fn(range_dims_per_task: TUPLE_OF_TWO_TUPLES) -> Callable:
    alphabet_sizes = [h - l for l, h in range_dims_per_task]
    def get_output_dim(task_id):
        """
        Here `task_id` is 0-indexed
        """
        return sum(alphabet_sizes[:(task_id + 1)])
    return get_output_dim
