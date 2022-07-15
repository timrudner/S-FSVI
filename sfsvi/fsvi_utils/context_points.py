"""
This file contains functions for generating context points by sampling from coreset or from sampling
randomly using context_point_fn.
"""
from typing import Callable, Dict

import numpy as np

from sfsvi.general_utils.jax_utils import KeyHelper
from benchmarking.benchmark_args import NOT_SPECIFIED
from sfsvi.fsvi_utils.coreset.coreset import Coreset


def make_context_points(
    not_use_coreset: bool,
    constant_context_points: bool,
    n_context_points: int,
    context_point_augmentation: bool,
    task_id: int,
    kh: KeyHelper,
    x_batch: np.ndarray,
    context_point_fn: Callable,
    coreset: Coreset,
    draw_per_class: bool = False,
    coreset_n_tasks = NOT_SPECIFIED,
    n_augment: str = None,
    augment_mode: str = "linear",
) -> Dict[int, np.ndarray]:
    x_context = _draw_context_points_for_previous_tasks(
        not_use_coreset=not_use_coreset,
        constant_context_points=constant_context_points,
        nb_previous_tasks=task_id,
        context_point_fn=context_point_fn,
        x_batch=x_batch,
        kh=kh,
        coreset=coreset,
        n_context_points=n_context_points,
        draw_per_class=draw_per_class,
        coreset_n_tasks=coreset_n_tasks,
    )
    if task_id == 0:
        n_augment = n_context_points
    x_context = _draw_context_points_for_current_task(
        x_context=x_context,
        task_id=task_id,
        context_point_augmentation=context_point_augmentation,
        n_augment=n_augment,
        augment_mode=augment_mode,
        context_point_fn=context_point_fn,
        x_batch=x_batch,
        kh=kh,
    )
    assert (
        max(list(x_context.keys())) <= task_id
    ), f"You defined context points for task {max(list(x_context.keys()))}, current task_id is {task_id}"
    return x_context


def _draw_context_points_for_current_task(
    x_context: Dict,
    task_id: int,
    context_point_augmentation: bool,
    n_augment: str,
    augment_mode: str,
    context_point_fn: Callable,
    x_batch: np.ndarray,
    kh: KeyHelper,
):
    if task_id == 0 or context_point_augmentation:
        # Appends a set of n_context_points input points to the set of context points
        assert task_id not in x_context, f"x_context[task_id]={x_context[task_id]}"
        n_augment = None if n_augment == NOT_SPECIFIED else int(n_augment)
        if n_augment is not None:
            if augment_mode == "constant":
                n_augment = n_augment
            elif augment_mode == "linear":
                n_augment = (task_id + 1) * n_augment
            else:
                raise NotImplementedError
        x_context[task_id] = context_point_fn(x_batch, kh.next_key(), n_augment)
    return x_context


def _draw_context_points_for_previous_tasks(
    not_use_coreset: bool,
    constant_context_points: bool,
    nb_previous_tasks: int,
    context_point_fn: Callable,
    x_batch: np.ndarray,
    kh: KeyHelper,
    coreset: Coreset,
    n_context_points: int,
    draw_per_class: bool,
    coreset_n_tasks,
):
    if not_use_coreset:
        # Draw context points according to random selection method
        if constant_context_points:
            if nb_previous_tasks == 0:
                x_context = {}
            else:
                context_points = context_point_fn(x_batch, kh.next_key())
                x_context = _even_distribute_per_task(
                    context_points, nb_previous_tasks
                )
        else:
            x_context = {
                t_id: context_point_fn(x_batch, kh.next_key())
                for t_id in range(nb_previous_tasks)
            }
    else:
        # Draw context points from context point buffer
        x_context = coreset.draw(
            n_context_points,
            draw_per_class=draw_per_class,
            coreset_n_tasks=coreset_n_tasks,
        )
    return x_context


def _even_distribute_per_task(
    context_points: np.ndarray, nb_previous_tasks: int
) -> Dict[int, np.ndarray]:
    n_total = len(context_points)
    n_points_per_task = n_total // nb_previous_tasks
    n_points_last_task = n_total - (nb_previous_tasks - 1) * n_points_per_task
    x_context = {
        i: context_points[i * n_points_per_task : (i + 1) * n_points_per_task]
        for i in range(nb_previous_tasks - 1)
    }
    x_context[nb_previous_tasks - 1] = context_points[-n_points_last_task:]
    assert sum([len(x) for x in x_context.values()]) == n_total
    return x_context
