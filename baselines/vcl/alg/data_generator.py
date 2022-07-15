from copy import deepcopy

import numpy as np

from baselines.vcl.alg import utils
from benchmarking.data_loaders.get_data import prepare_data


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


class CustomGenerator:
    def __init__(self, task, use_val_split, n_permuted_tasks):
        (
            self.load_task,
            meta_data,
        ) = prepare_data(
            task, use_val_split, n_permuted_tasks=n_permuted_tasks
        )
        self.n_tasks = meta_data["n_tasks"]
        self.input_shape = meta_data["input_shape"]
        self.output_dim = meta_data["output_dim"]

        self.use_val_split = use_val_split
        self.task_id = 0
        self.cur_iter = 0
        self.max_iter = self.n_tasks

    def get_dims(self):
        # Get data input and output dimensions
        return self.input_shape[1], self.output_dim

    def next_task(self):
        d = self._get_task(task_id=self.cur_iter)
        self.cur_iter += 1
        return d

    def _get_task(self, task_id):
        data = self.load_task(task_id=task_id, return_tfds=False)
        next_x_train, next_y_train, next_x_test, next_y_test = data["train"] + data["test"]
        next_y_train = _one_hot(np.array(next_y_train), self.output_dim)
        next_y_test = _one_hot(np.array(next_y_test), self.output_dim)
        return next_x_train, next_y_train, next_x_test, next_y_test


def is_multihead(task):
    return is_smnist_multihead(task) or "sfashionmnist" in task


def is_smnist_multihead(task):
    return "smnist" in task and "smnist_sh" not in task


def is_single_head(task):
    return "pmnist" in task or "pfashionmnist" in task or "smnist_sh" in task or "sfashionmnist_sh" in task


class SplitMnistGenerator:
    def __init__(self):
        train_set, valid_set, test_set = utils.load_original_data()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack(
                (self.X_train[train_0_id], self.X_train[train_1_id])
            )

            next_y_train = np.vstack(
                (np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1)))
            )
            next_y_train = np.hstack((next_y_train, 1 - next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack(
                (np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1)))
            )
            next_y_test = np.hstack((next_y_test, 1 - next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


class PermutedMnistGenerator:
    def __init__(self, max_iter=10):
        train_set, valid_set, test_set = utils.load_original_data()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:, perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
