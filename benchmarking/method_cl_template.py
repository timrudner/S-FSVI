"""This file provides a template for implementing a continual learning method
that can be tested by `ContinualLearningProtocol`."""
from logging import Logger
from typing import Callable
from typing import Dict
from typing import Optional

import tensorflow as tf


class MethodCLTemplate:
    def run_one_task(
        self,
        task_id: int,
        train_dataset: tf.data.Dataset,
        n_train: int,
        validation_dataset: Optional[tf.data.Dataset],
        n_valid: Optional[int],
        fn_for_logging_info_per_epoch: Callable[[Callable, int], None],
        logger: Logger,
    ) -> Dict:
        """Train model on one task and evaluate the model on tasks seen so far.

        :param task_id: (0-indexed) task number, for example, the second task
            has a task_id of 1.
        :param train_dataset: training dataset that has been infinitely
            repeated but not batched.
        :param n_train: number of samples in the training set.
        :param validation_dataset: validation dataset that has been infinitely
            repeated but not batched, optional.
        :param n_valid: number of samples in the validation set.
        :param fn_for_logging_info_per_epoch: a Callable for evaluating
            the model at the end of each epoch, it takes in a function for
            predicting class probabilities and the number of epochs, evaluates
            the method on all tasks seen so far. The results are saved and
            displayed.
        :param logger: a logger.
        :return
            data to be saved as part of the artifact.
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
