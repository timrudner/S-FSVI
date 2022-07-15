from typing import Dict

import numpy as np

from benchmarking.benchmark_args import NOT_SPECIFIED
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class


class Coreset:
    """
    This class is responsible for managing the context points for each task.

    self.task_coresets is a list of numpy arrays
    """

    def __init__(self):
        self.task_coresets = []
        self.task_coreset_labels = []

    def wrap_data_to_save(self, latest_task: bool):
        return {
            "x": self.task_coresets[-1]
            if latest_task and self.task_coresets
            else self.task_coresets,
            "y": self.task_coreset_labels[-1]
            if latest_task and self.task_coreset_labels
            else self.task_coreset_labels,
        }

    def add_coreset_points(self, x_candidate, y_candidate, inds_add):
        self.task_coresets.append(x_candidate[inds_add])
        self.task_coreset_labels.append(y_candidate[inds_add])

    def draw(
            self,
            n_draw: int,
            draw_per_class: bool = False,
            coreset_n_tasks: str = NOT_SPECIFIED,
        ) -> Dict[int, np.ndarray]:
        """
        Draw [n_draw] samples from each array in the self.task_coresets

        @param n_draw: the number of points to draw from the coreset for each task
        """
        if draw_per_class:
            return self.draw_per_class(n_draw)
        n_tasks = len(self.task_coresets)
        if coreset_n_tasks != NOT_SPECIFIED and n_tasks > int(coreset_n_tasks):
            # Use a subset of task coresets
            task_subset = np.random.choice(n_tasks, size=int(coreset_n_tasks), replace=False)
            coresets = [self.task_coresets[i] for i in task_subset]
        else:
            # Use all task coresets
            coresets = self.task_coresets
        x_draw = {}
        for task_id, x_coreset in enumerate(coresets):
            n_choice = len(x_coreset)
            if n_choice < n_draw:
                print(f"The number of available coreset points is {n_choice}, "
                      f"smaller than the number of points to draw {n_draw}")
                n_draw = n_choice
            inds_draw = np.random.choice(n_choice, size=n_draw, replace=False)
            x_draw[task_id] = x_coreset[inds_draw]
        return x_draw

    def draw_per_class(self, n_draw: int) -> Dict[int, np.ndarray]:
        """
        Draw [n_draw] samples from each array in the self.task_coresets

        @param n_draw: the number of points to draw from the coreset for each task
        """
        x_draw = {}
        for task_id, (x, y) in enumerate(
            zip(self.task_coresets, self.task_coreset_labels)
        ):
            inds_draw = add_by_random_per_class(y_candidate=y, n_add=n_draw)
            x_draw[task_id] = x[inds_draw]
        return x_draw

    def draw_unbalanced(self, budget: int) -> Dict[int, np.ndarray]:
        """[Unused/untested] Sample a different number of points from each task coreset."""
        n_tasks = len(self.task_coresets)
        n_draw = budget * np.random.dirichlet(np.ones(n_tasks - 1))
        n_draw = n_draw.astype(int).tolist()
        n_draw.append(budget - sum(n_draw))
        x_draw = {}
        for i, (x_coreset_i, n_draw_i) in enumerate(zip(self.task_coresets, n_draw)):
            n_choice = len(x_coreset_i)
            inds_draw = np.random.choice(n_choice, size=n_draw_i, replace=False)
            x_draw[i] = x_coreset_i[inds_draw]
        return x_draw