import numpy as np


def get_model_head_settings(
    n_tasks: int,
    n_classes: int,
    head: str,
):
    """
    :param head: either "single" or "multi"
    @return:
            range_dims_per_task: a sequence of 2-tuples, each 2-tuple (start, end) corresponds to one task,
                    the output head is from `start` till `end` (exclusive)
            max_entropies: a sequence of floats, each value corresponds to the maximum possible entropy
                    produced by the prediction
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
