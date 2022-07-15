"""Utilities for generating toy data."""
import numpy as np
import sklearn
import torch
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset

from benchmarking.data_loaders.utils import get_model_head_settings


def get_toy_data(
    dataset: str,
    use_val_split: bool,
    shuffle_seed=None,
    generator_seed=None,
    no_shuffle=False,
    **kwargs,
):
    """Returns data and meta data for the 2-dimensional toy task sequence."""
    test_split_size = 0.1
    nb_classes_per_task = 2
    assert not use_val_split, "Not supporting use_val_split=True"
    generator = ToydataGenerator(random_state=generator_seed)
    n_tasks = generator.max_iter
    adjust_y_offset = "toy_reprod" not in dataset

    data = []
    for task_id in range(n_tasks):
        x_train, y_train = generator.get_task_by_id(
            task_id=task_id, return_np_array=True
        )
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x_train,
            y_train,
            test_size=test_split_size,
            random_state=shuffle_seed,
            shuffle=not no_shuffle,
        )
        if adjust_y_offset:
            y_train = y_train + nb_classes_per_task * task_id
            y_test = y_test + nb_classes_per_task * task_id
        data.append(
            {
                "train": [x_train, y_train],
                "test": [x_test, y_test],
            }
        )

    head = "single" if dataset in {"toy_sh", "toy_reprod"} else "multi"
    output_dim = 2 * n_tasks if adjust_y_offset else 2
    range_dims_per_task, max_entropies = get_model_head_settings(
        n_tasks=n_tasks,
        n_classes=output_dim,
        head=head,
    )
    meta_data = {
        "n_tasks": n_tasks,
        "n_train": generator.total_n_train,
        "input_shape": [1, 2],
        "output_dim": output_dim,
        "n_coreset_inputs_per_task_list": n_tasks * (40,),
        "range_dims_per_task": range_dims_per_task,
        "max_entropies": max_entropies,
    }
    return data, meta_data


def _adjust_y(
    y: np.ndarray,
    task_id: int,
    nb_classes_per_task: int,
) -> np.ndarray:
    return y + nb_classes_per_task * task_id


class ToydataGenerator:
    """
    Generate binary classification tasks.
    """

    def __init__(self, max_iter=5, num_samples=2000, option=0, random_state=None):

        self.offset = 5  # Offset when loading data in next_task()

        # Generate data
        if option == 0:
            # Standard settings
            centers = [
                [0, 0.2],
                [0.6, 0.9],
                [1.3, 0.4],
                [1.6, -0.1],
                [2.0, 0.3],
                [0.45, 0],
                [0.7, 0.45],
                [1.0, 0.1],
                [1.7, -0.4],
                [2.3, 0.1],
            ]
            std = [
                [0.08, 0.22],
                [0.24, 0.08],
                [0.04, 0.2],
                [0.16, 0.05],
                [0.05, 0.16],
                [0.08, 0.16],
                [0.16, 0.08],
                [0.06, 0.16],
                [0.24, 0.05],
                [0.05, 0.22],
            ]

        elif option == 1:
            # Six tasks
            centers = [
                [0, 0.2],
                [0.6, 0.9],
                [1.3, 0.4],
                [1.6, -0.1],
                [2.0, 0.3],
                [1.65, 0.1],
                [0.45, 0],
                [0.7, 0.45],
                [1.0, 0.1],
                [1.7, -0.4],
                [2.3, 0.1],
                [0.7, 0.25],
            ]
            std = [
                [0.08, 0.22],
                [0.24, 0.08],
                [0.04, 0.2],
                [0.16, 0.05],
                [0.05, 0.16],
                [0.14, 0.14],
                [0.08, 0.16],
                [0.16, 0.08],
                [0.06, 0.16],
                [0.24, 0.05],
                [0.05, 0.22],
                [0.14, 0.14],
            ]

        elif option == 2:
            # All std devs increased
            centers = [
                [0, 0.2],
                [0.6, 0.9],
                [1.3, 0.4],
                [1.6, -0.1],
                [2.0, 0.3],
                [0.45, 0],
                [0.7, 0.45],
                [1.0, 0.1],
                [1.7, -0.4],
                [2.3, 0.1],
            ]
            std = [
                [0.12, 0.22],
                [0.24, 0.12],
                [0.07, 0.2],
                [0.16, 0.08],
                [0.08, 0.16],
                [0.12, 0.16],
                [0.16, 0.12],
                [0.08, 0.16],
                [0.24, 0.08],
                [0.08, 0.22],
            ]

        elif option == 3:
            # Tougher to separate
            centers = [
                [0, 0.2],
                [0.6, 0.65],
                [1.3, 0.4],
                [1.6, -0.22],
                [2.0, 0.3],
                [0.45, 0],
                [0.7, 0.55],
                [1.0, 0.1],
                [1.7, -0.3],
                [2.3, 0.1],
            ]
            std = [
                [0.08, 0.22],
                [0.24, 0.08],
                [0.04, 0.2],
                [0.16, 0.05],
                [0.05, 0.16],
                [0.08, 0.16],
                [0.16, 0.08],
                [0.06, 0.16],
                [0.24, 0.05],
                [0.05, 0.22],
            ]

        elif option == 4:
            # Two tasks, of same two gaussians
            centers = [[0, 0.2], [0, 0.2], [0.45, 0], [0.45, 0]]
            std = [[0.08, 0.22], [0.08, 0.22], [0.08, 0.16], [0.08, 0.16]]

        else:
            # If new / unknown option
            centers = [
                [0, 0.2],
                [0.6, 0.9],
                [1.3, 0.4],
                [1.6, -0.1],
                [2.0, 0.3],
                [0.45, 0],
                [0.7, 0.45],
                [1.0, 0.1],
                [1.7, -0.4],
                [2.3, 0.1],
            ]
            std = [
                [0.08, 0.22],
                [0.24, 0.08],
                [0.04, 0.2],
                [0.16, 0.05],
                [0.05, 0.16],
                [0.08, 0.16],
                [0.16, 0.08],
                [0.06, 0.16],
                [0.24, 0.05],
                [0.05, 0.22],
            ]

        if option != 1 and max_iter > 5:
            raise Exception("Current toydatagenerator only supports up to 5 tasks.")

        self.X, self.y = make_blobs(
            num_samples * 2 * max_iter,
            centers=centers,
            cluster_std=std,
            random_state=random_state,
        )
        self.X = self.X.astype("float32")
        h = 0.01
        self.x_min, self.x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        self.y_min, self.y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2
        self.data_min = np.array([self.x_min, self.y_min], dtype="float32")
        self.data_max = np.array([self.x_max, self.y_max], dtype="float32")
        self.data_min = np.expand_dims(self.data_min, axis=0)
        self.data_max = np.expand_dims(self.data_max, axis=0)
        xx, yy = np.meshgrid(
            np.arange(self.x_min, self.x_max, h), np.arange(self.y_min, self.y_max, h)
        )
        xx = xx.astype("float32")
        yy = yy.astype("float32")
        self.test_shape = xx.shape
        X_test = np.c_[xx.ravel(), yy.ravel()]
        self.X_test = torch.from_numpy(X_test)
        self.y_test = torch.zeros((len(self.X_test)), dtype=self.X_test.dtype)
        self.max_iter = max_iter
        self.num_samples = num_samples  # number of samples per task

        if option == 1:
            self.offset = 6
        elif option == 4:
            self.offset = 2

        self.cur_iter = 0

    @property
    def total_n_train(self):
        return len(self.X)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            data = self.get_task_by_id(task_id=self.cur_iter)
            self.cur_iter += 1
            return data

    def get_task_by_id(self, task_id: int, return_np_array=False):
        x_train_0 = self.X[self.y == task_id]
        x_train_1 = self.X[self.y == task_id + self.offset]
        y_train_0 = np.zeros_like(self.y[self.y == task_id])
        y_train_1 = np.ones_like(self.y[self.y == task_id + self.offset])
        x_train = np.concatenate([x_train_0, x_train_1], axis=0)
        y_train = np.concatenate([y_train_0, y_train_1], axis=0)
        y_train = y_train.astype("int64")
        if return_np_array:
            return x_train, y_train
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        return TensorDataset(x_train, y_train)

    def reset(self):
        self.cur_iter = 0
