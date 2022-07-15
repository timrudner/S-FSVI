from typing import Tuple

import numpy as np


def get_model_head_settings(
    n_tasks: int,
    n_classes: int,
    head: str,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...]]:
    """Return the range of dimension for the output heads of each task, as
    well as the maximum entropy for each task.

    :param n_tasks: number tasks.
    :param n_classes: total number of classes.
    :param head: either "single" or "multi".
    :return:
        range_dims_per_task: output heads index range for each task.
                For example, for split MNIST (MH), this variable is
                    `range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
                which means output heads for the first task are the 1st and 2nd
                output dimensions, the output heads for the second task are the
                3rd and 4th dimension, etc.
        max_entropies: each value corresponds to the maximum possible entropy
            produced by the prediction.
    """
    if head == "single":
        one_task_range = (
            0,
            n_classes,
        )
        range_dims_per_task = n_tasks * (one_task_range,)
        max_entropies = n_tasks * (-np.log(1 / n_classes),)
    else:
        head_size = n_classes // n_tasks
        range_dims_per_task = tuple(
            zip(
                range(0, n_classes, head_size),
                range(head_size, n_classes + 1, head_size),
            )
        )
        max_entropies = n_tasks * (-np.log(1 / head_size),)
    return range_dims_per_task, max_entropies
