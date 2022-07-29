from typing import Callable
from typing import Iterator
from typing import Dict


class MethodCLTemplate:
    def run_one_task(
        self,
        task_id: int,
        train_iterator: Iterator,
        n_train: int,
        fn_for_logging_info_per_epoch: Callable[[Callable, int], None],
    ) -> None:
        """Train model on one task and evaluate the model on tasks seen so far.

        :param task_id: (0-indexed) task index, e.g. task_id = 0 for the first
            task.
        :param train_iterator: an iterator for dataset, next(train_iterator)
            yields a pair of (x, y), where x is one image and y is its label.
            The iterator has infinite length because the corresponding tfds
            dataset was created with `.repeat()`.
        :param n_train: the total number of training samples per epoch.
        :param fn_for_logging_info_per_epoch: a function that takes in
            two arguments:
                pred_fn: a function for making prediction, it takes in the
                    task_id and a batch of input images and returns a batch
                    of predictions.
                epoch: the epoch number
            This function is used for evaluating the method on tasks seen so
            far.
        :return:
        """
        # ... given the data of this task, train the model ...
        # ... every few epochs, call `fn_for_logging_info_per_epoch` on
        #   your model to evaluate it on all tasks seen so far ...
        raise NotImplementedError

    def get_final_data_to_log(self) -> Dict:
        """Return final states to be saved as part of the artifacts.

        :return:
            data to be saved.
        """
        return {}
