"""Class for storing and sampling from coreset."""
from typing import Dict

import numpy as np
import jax.numpy as jnp

from benchmarking.benchmark_args import NOT_SPECIFIED
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class


class Coreset:
    """This is responsible for storing the coreset points and sampling
    context points from them."""

    def __init__(self):
        self.task_coresets = []
        self.task_coreset_labels = []

    def wrap_data_to_save(self, latest_task: bool) -> Dict[str, jnp.ndarray]:
        """Prepare coreset points to save."""
        return {
            "x": self.task_coresets[-1]
            if latest_task and self.task_coresets
            else self.task_coresets,
            "y": self.task_coreset_labels[-1]
            if latest_task and self.task_coreset_labels
            else self.task_coreset_labels,
        }

    def add_coreset_points(
        self,
        x_candidate: jnp.ndarray,
        y_candidate: jnp.ndarray,
        inds_add: np.ndarray,
    ) -> None:
        self.task_coresets.append(x_candidate[inds_add])
        self.task_coreset_labels.append(y_candidate[inds_add])

    def draw(
        self,
        n_draw: int,
        draw_per_class: bool = False,
        coreset_n_tasks: str = NOT_SPECIFIED,
    ) -> Dict[int, np.ndarray]:
        """Draw `n_draw` context points from the stored coreset points.

        :param n_draw: the number of points to draw from the coreset for each
            task.
        :param draw_per_class: if True, draw equal number of context points of
            different classes from the coreset.
        :param coreset_n_tasks: if the value is not equal to `NOT_SPECIFIED`, then
            only draw context points from a random subset of `int(coreset_n_tasks)`
            tasks.
        :returns
            a mapping from task id to coreset points.
        """
        if draw_per_class:
            return self.draw_per_class(n_draw)
        n_tasks = len(self.task_coresets)
        if coreset_n_tasks != NOT_SPECIFIED and n_tasks > int(coreset_n_tasks):
            # Use a subset of task coresets
            task_subset = np.random.choice(
                n_tasks, size=int(coreset_n_tasks), replace=False
            )
            coresets = [self.task_coresets[i] for i in task_subset]
        else:
            # Use all task coresets
            coresets = self.task_coresets
        x_draw = {}
        for task_id, x_coreset in enumerate(coresets):
            n_choice = len(x_coreset)
            if n_choice < n_draw:
                print(
                    f"The number of available coreset points is {n_choice}, "
                    f"smaller than the number of points to draw {n_draw}"
                )
                n_draw = n_choice
            inds_draw = np.random.choice(n_choice, size=n_draw, replace=False)
            x_draw[task_id] = x_coreset[inds_draw]
        return x_draw

    def draw_per_class(self, n_draw: int) -> Dict[int, np.ndarray]:
        """Draw `n_draw` context points from the stored coreset points with
        equal amount of points from each class.

        :param n_draw: the number of points to draw from the coreset for each
            task.
        :returns
            a mapping from task id to coreset points.
        """
        x_draw = {}
        for task_id, (x, y) in enumerate(
            zip(self.task_coresets, self.task_coreset_labels)
        ):
            inds_draw = add_by_random_per_class(y_candidate=y, n_add=n_draw)
            x_draw[task_id] = x[inds_draw]
        return x_draw
