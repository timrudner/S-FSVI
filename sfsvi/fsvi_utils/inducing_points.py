"""
This file contains functions for generating inducing points by sampling from coreset or from sampling
randomly using inducing_input_fn.
"""
from typing import Callable, Dict

import numpy as np

from sfsvi.general_utils.jax_utils import KeyHelper
from sfsvi.fsvi_utils.args_cl import NOT_SPECIFIED
from sfsvi.fsvi_utils.coreset.coreset import Coreset


def make_inducing_points(
    not_use_coreset: bool,
    constant_inducing_points: bool,
    n_inducing_inputs: int,
    inducing_input_augmentation: bool,
    task_id: int,
    kh: KeyHelper,
    x_batch: np.ndarray,
    inducing_input_fn: Callable,
    coreset: Coreset,
    draw_per_class: bool = False,
    coreset_n_tasks = NOT_SPECIFIED,
    n_augment: str = None,
    augment_mode: str = "linear",
) -> Dict[int, np.ndarray]:
    x_inducing = _draw_inducing_inputs_for_previous_tasks(
        not_use_coreset=not_use_coreset,
        constant_inducing_points=constant_inducing_points,
        nb_previous_tasks=task_id,
        inducing_input_fn=inducing_input_fn,
        x_batch=x_batch,
        kh=kh,
        coreset=coreset,
        n_inducing_inputs=n_inducing_inputs,
        draw_per_class=draw_per_class,
        coreset_n_tasks=coreset_n_tasks,
    )
    if task_id == 0:
        n_augment = n_inducing_inputs
    x_inducing = _draw_inducing_inputs_for_current_task(
        x_inducing=x_inducing,
        task_id=task_id,
        inducing_input_augmentation=inducing_input_augmentation,
        n_augment=n_augment,
        augment_mode=augment_mode,
        inducing_input_fn=inducing_input_fn,
        x_batch=x_batch,
        kh=kh,
    )
    assert (
        max(list(x_inducing.keys())) <= task_id
    ), f"You defined inducing points for task {max(list(x_inducing.keys()))}, current task_id is {task_id}"
    return x_inducing


def _draw_inducing_inputs_for_current_task(
    x_inducing: Dict,
    task_id: int,
    inducing_input_augmentation: bool,
    n_augment: str,
    augment_mode: str,
    inducing_input_fn: Callable,
    x_batch: np.ndarray,
    kh: KeyHelper,
):
    if task_id == 0 or inducing_input_augmentation:
        # Appends a set of n_inducing_inputs input points to the set of inducing inputs
        assert task_id not in x_inducing, f"x_inducing[task_id]={x_inducing[task_id]}"
        n_augment = None if n_augment == NOT_SPECIFIED else int(n_augment)
        if n_augment is not None:
            if augment_mode == "constant":
                n_augment = n_augment
            elif augment_mode == "linear":
                n_augment = (task_id + 1) * n_augment
            else:
                raise NotImplementedError
        x_inducing[task_id] = inducing_input_fn(x_batch, kh.next_key(), n_augment)
    return x_inducing


def _draw_inducing_inputs_for_previous_tasks(
    not_use_coreset: bool,
    constant_inducing_points: bool,
    nb_previous_tasks: int,
    inducing_input_fn: Callable,
    x_batch: np.ndarray,
    kh: KeyHelper,
    coreset: Coreset,
    n_inducing_inputs: int,
    draw_per_class: bool,
    coreset_n_tasks,
):
    if not_use_coreset:
        # Draw inducing inputs according to random selection method
        if constant_inducing_points:
            if nb_previous_tasks == 0:
                x_inducing = {}
            else:
                inducing_points = inducing_input_fn(x_batch, kh.next_key())
                x_inducing = _even_distribute_per_task(
                    inducing_points, nb_previous_tasks
                )
        else:
            x_inducing = {
                t_id: inducing_input_fn(x_batch, kh.next_key())
                for t_id in range(nb_previous_tasks)
            }
    else:
        # Draw inducing inputs from inducing input buffer
        x_inducing = coreset.draw(
            n_inducing_inputs,
            draw_per_class=draw_per_class,
            coreset_n_tasks=coreset_n_tasks,
        )
    return x_inducing


def _even_distribute_per_task(
    inducing_points: np.ndarray, nb_previous_tasks: int
) -> Dict[int, np.ndarray]:
    n_total = len(inducing_points)
    n_points_per_task = n_total // nb_previous_tasks
    n_points_last_task = n_total - (nb_previous_tasks - 1) * n_points_per_task
    x_inducing = {
        i: inducing_points[i * n_points_per_task : (i + 1) * n_points_per_task]
        for i in range(nb_previous_tasks - 1)
    }
    x_inducing[nb_previous_tasks - 1] = inducing_points[-n_points_last_task:]
    assert sum([len(x) for x in x_inducing.values()]) == n_total
    return x_inducing
