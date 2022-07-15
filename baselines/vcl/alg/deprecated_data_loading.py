from functools import partial
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np

from sfsvi.fsvi_utils.utils_cl import TUPLE_OF_TWO_TUPLES
from benchmarking.data_loaders.mnist_and_cifar import \
    select_task_examples
from sfsvi.fsvi_utils.datasets_cl import DatasetSplit
from sfsvi.fsvi_utils.datasets_cl import RawData
from sfsvi.fsvi_utils.datasets_cl import get_model_head_settings
from sfsvi.fsvi_utils.datasets_cl import load_raw_mnist


def prepare_data_vcl(
    task: str,
    use_val_split=True,
    fix_shuffle=False,
    n_permuted_tasks: int = 10,
) -> Tuple[
    Callable[[], DatasetSplit],
    int,
    int,
    Tuple[int],
    List[int],
    int,
    TUPLE_OF_TWO_TUPLES,
    Tuple[float, ...],
]:
    """
    Returns various dataset information for a specific continual learning set up.

    @param task: determine the type of task
    @param fix_shuffle: whether to fix the random seed of shuffle of training set.
    @param use_val_split: determines the size of training dataset
    @param n_permuted_tasks: number of permuted tasks
    @return:
        load_task_partial: the function that takes in task id and returns a DatasetSplit object
        n_tasks: the number of tasks
        n_train: the number of training samples
        n_coreset_inputs_per_task_list: a tuple of integers, the number of inducing points for each task
        input_shape: the shape of input data with batch dimension to be 1
        output_dim: output dimension of task
        range_dims_per_task: a sequence of 2-tuples, each 2-tuple (start, end) corresponds to one task,
            the output head is from `start` till `end` (exclusive)
        max_entropies: a sequence of floats, each value corresponds to the maximum possible entropy
            produced by the prediction
    """
    # `n_inducing_inputs_per_task` = number of inducing points saved to coreset per task
    # TODO: check `n_train` for split tasks (can affect selection of inducing inputs)
    if "mnist" in task:
        if ("pmnist" in task) or ("pfashionmnist" in task):
            task_mode = "permuted"
            n_tasks = n_permuted_tasks
            n_coreset_inputs_per_task = 200
        elif ("smnist" in task) or ("sfashionmnist" in task):
            task_mode = "split"
            n_tasks = 5
            n_coreset_inputs_per_task = (
                200 if "_sh" in task else 40
            )  #  more for single-head setup
        else:
            raise ValueError("Unrecognized task.")
        mnist_mode = "fashion" if "fashion" in task else "standard"
        # TODO: minor, remove hardcoded value
        n_train = 50000 if use_val_split else 60000
        input_shape = [1, 784]
        output_dim = 10
        use_data_augmentation = False
        raw_data = load_raw_mnist(
            n_tasks=n_tasks,
            fix_shuffle=fix_shuffle,
            mnist_mode=mnist_mode,
            task_mode=task_mode,
        )
    else:
        raise ValueError("Unrecognized task.")
    load_task_partial = partial(
        load_task_vcl,
        raw_data=raw_data,
        use_val_split=use_val_split,
        task_mode=task_mode,
    )
    n_coreset_inputs_per_task_list = n_tasks * (n_coreset_inputs_per_task,)
    range_dims_per_task, max_entropies = get_model_head_settings(
        task, n_tasks, output_dim, raw_data
    )
    return (
        load_task_partial,
        n_tasks,
        n_train,
        n_coreset_inputs_per_task_list,
        input_shape,
        output_dim,
        range_dims_per_task,
        max_entropies,
    )


def load_task_vcl(
    raw_data: RawData, task_id: int, use_val_split: bool, task_mode: str = "permuted",
):
    """
    1. Split entire data into train, valid, test sets
    2. Process data according to task id
        for permuted task, permute data dimension
        for split task, select subset of samples corresponding to the task

    @param use_val_split: if true, split a validation set of the same size as test set from the training set
    """
    assert not use_val_split
    if task_mode == "permuted":
        task_permutations, _, n_test = raw_data.task_metadata
        x_train, x_test = np.split(raw_data.inputs, (-n_test,))
        y_train, y_test = np.split(raw_data.outputs, (-n_test,))
        x_train = x_train[:, task_permutations[task_id]]
        x_test = x_test[:, task_permutations[task_id]]
    elif task_mode == "split":
        task_classes, n_test = raw_data.task_metadata
        x_train, x_test = np.split(raw_data.inputs, (-n_test,))
        y_train, y_test = np.split(raw_data.outputs, (-n_test,))
        x_train, y_train = select_task_examples(x_train, y_train, task_classes[task_id])
        x_test, y_test = select_task_examples(x_test, y_test, task_classes[task_id])
    else:
        raise ValueError("Unrecognized task mode.")
    return x_train, y_train, x_test, y_test
