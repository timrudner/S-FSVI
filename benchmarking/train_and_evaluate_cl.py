"""This file contains a class for training and evaluating a continual learning
method on a task sequence."""
from copy import copy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from benchmarking.benchmark_args import NOT_SPECIFIED
from benchmarking.data_loaders.get_data import DatasetSplit
from benchmarking.data_loaders.get_data import make_iterators
from benchmarking.data_loaders.get_data import prepare_data
from benchmarking.method_cl_template import MethodCLTemplate
from sfsvi.fsvi_utils.utils_cl import TrainingLog
from sfsvi.fsvi_utils.utils_cl import evaluate_on_all_tasks
from sfsvi.general_utils.log import Hyperparameters
from sfsvi.general_utils.log import create_logdir
from sfsvi.general_utils.log import save_chkpt
from sfsvi.general_utils.log import save_kwargs
from sfsvi.general_utils.log import set_up_logging


class ContinualLearningProtocol:
    """
    This class is responsible for training and evaluating a continual learning
    method on a task sequence.
    """

    def __init__(
        self,
        orig_cmd: Optional[str] = None,
        input_type=np.float32,
        **kwargs,
    ):
        """
        :param orig_cmd: the original command line arguments, if provided, this
            command is saved in the `command_line.txt` artifact of the
            experiment instead of `sys.argv`.
        :param input_type: the data type of input images.
        :param kwargs: hyperparameters for the continual learning protocol.
        """
        self.initialize(orig_cmd, input_type, **kwargs)
        self.cl_method = None

    def train(self, cl_method: MethodCLTemplate) -> str:
        """Train and evaluate a continual learning method on a task sequence.

        :param cl_method: a continual learning method inplementing the
            `MethodCLTemplate` interface.
        :return:
            the path to the artifacts of the experiment.
        """
        self.cl_method = cl_method
        for task_id in range(self.n_tasks):
            self.run_one_task(task_id=task_id)
        self._logging_info_final()
        return self.logdir

    def run_one_task(self, task_id: int) -> None:
        """Train a continual learning model on one task.

        :param task_id: 0-based task identifier.
        """
        # prepare data
        task_data = self._prepare_data_per_task(task_id=task_id)
        # fit the model
        training_log = TrainingLog(task_id)
        n_train = (
            task_data.n_train
            if self.hparams.debug_n_train is None
            else self.hparams.debug_n_train
        )
        info_to_save = self.cl_method.run_one_task(
            task_id=task_id,
            train_dataset=task_data.train,
            n_train=n_train,
            validation_dataset=task_data.valid,
            n_valid=task_data.n_valid,
            fn_for_logging_info_per_epoch=self._get_fn_for_logging_info_per_epoch(
                training_log
            ),
            logger=self.logger,
        )
        # save the logged data
        self._logging_info_per_task(
            training_log=training_log,
            task_id=task_id,
            info_to_save=info_to_save,
        )

    def _get_fn_for_logging_info_per_epoch(
        self,
        training_log: TrainingLog,
    ) -> Callable:
        """
        The reason to declare the logging function here is to avoid the CLMethod to
        know about the metrics computed in this training script.

        :param training_log: a TrainingLog object for storing and displaying
            performance metrics.
        :return
            a function that will be passed to the continual learning method
            for evaluating the method on all tasks seen so far. The function
            takes in a function for predicting class probabilities and the
            number of epochs, evaluates the method on all tasks seen so
            far, collect, and display the results.
        """

        def evaluate_pred_fn(
            pred_fn: Callable,
            epoch: int,
        ) -> None:
            """Evaluate a model on tasks seen so far, collect and display
            metrics for each epoch.

            :param pred_fn: a function takes in two keyword arguments:
                    x: `jnp.ndarray`, input images to the model.
                    task_id: `int`, 0-based task identifier.
                and returns a `jnp.ndarray` which is the predicted probabilities
                of different classes.
            :param epoch: this can be removed once unused code is removed.
            :return:
            """
            (
                accuracies,
                entropies,
                accuracies_test,
                entropies_test,
            ) = self.evaluate_model(pred_fn)

            to_log = {
                "accuracies_test": accuracies_test,
                "entropies_test": entropies_test,
                "epoch": epoch,
            }
            if self.hparams.use_val_split:
                to_log.update(
                    {
                        "accuracies_valid": accuracies,
                        "entropies_valid": entropies,
                    }
                )
            training_log.update(**to_log)
            training_log.print_progress(epoch == 0)

        return evaluate_pred_fn

    def _logging_info_final(self):
        """Save information after the method is trained on all tasks."""
        training_log_df = pd.concat(self.training_log_dataframes)
        data = self.cl_method.get_final_data_to_log()
        # save_chkpt is part of new logging system
        if not self.hparams.no_artifact:
            save_chkpt(
                p=self.logdir / "chkpt",
                hparams=self.hparams.as_dict(),
                training_log_df=training_log_df,
                **data,
            )

    def _logging_info_per_task(
        self,
        training_log: TrainingLog,
        task_id: int,
        info_to_save: Dict,
    ) -> None:
        """Save information after training the method on each task."""
        training_log.print_short()
        self.training_log_dataframes.append(training_log.get_dataframe(self.n_tasks))

        if self.hparams.save_alt:
            training_log.save_task_specific_log(
                save_path=self.logdir / "raw_training_log" / str(task_id),
                n_tasks=self.n_tasks,
                task_id=task_id,
                hparams=self.hparams.as_dict(),
                **info_to_save,
            )

    def initialize(
        self,
        orig_cmd: str = None,
        input_type=np.float32,
        **kwargs,
    ) -> None:
        """Initialise logging, prepare data and hyperparameters."""
        # initialize logging
        _process_kwargs_protocol(kwargs)
        self.hparams = Hyperparameters(**kwargs)
        if self.hparams.no_artifact:
            self.logdir = None
            self.logger = None
        else:
            self.logdir = create_logdir(
                self.hparams.logroot, self.hparams.subdir, cmd=orig_cmd
            )
            self.logger = set_up_logging(log_path=self.logdir / "log")

        if self.logdir:
            save_kwargs(kwargs=self.hparams.as_dict(), path=self.logdir / "kwargs")

        # Number of coreset points is an important factor influencing the
        # performance, so it is determinted by this training script.
        (self.load_task, meta_data) = prepare_data(
            task=self.hparams.task,
            use_val_split=self.hparams.use_val_split,
            n_permuted_tasks=self.hparams.n_permuted_tasks,
            n_omniglot_tasks=self.hparams.n_omniglot_tasks,
            n_valid=self.hparams.n_valid,
            fix_shuffle=self.hparams.fix_shuffle,
            n_omniglot_coreset_chars=self.hparams.n_omniglot_coreset_chars,
            omniglot_test_random_state=self.hparams.seed
            if self.hparams.omniglot_randomize_test_split
            else 0,
            omniglot_randomize_task_sequence_seed=self.hparams.seed
            if self.hparams.omniglot_randomize_task_sequence
            else None,
            input_dtype=input_type,
        )
        self.n_tasks = meta_data["n_tasks"]
        self.n_train = meta_data["n_train"]
        self.n_coreset_inputs_per_task_list = meta_data[
            "n_coreset_inputs_per_task_list"
        ]
        self.input_shape = meta_data["input_shape"]
        self.output_dim = meta_data["output_dim"]
        self.range_dims_per_task = meta_data["range_dims_per_task"]
        self.max_entropies = meta_data["max_entropies"]

        if self.hparams.n_coreset_inputs_per_task != NOT_SPECIFIED:
            self.n_coreset_inputs_per_task_list = [
                int(self.hparams.n_coreset_inputs_per_task)
            ] * self.n_tasks

        (
            self.training_log_dataframes,
            self.valid_iterators,
            self.test_iterators,
        ) = ([], {}, {})

    def _prepare_data_per_task(self, task_id: int) -> DatasetSplit:
        """Load data for one task."""
        task_data = self.load_task(task_id=task_id)
        task_iterators = make_iterators(task_data, self.hparams.batch_size)
        if task_iterators.full_valid is not None:
            self.valid_iterators[task_id] = task_iterators.full_valid
        self.test_iterators[task_id] = task_iterators.full_test
        return task_data

    def evaluate_model(
        self,
        pred_fn: Callable,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Evaluate model on test set and validation set (if there is any)."""
        accuracies, entropies = evaluate_on_all_tasks(
            self.valid_iterators, pred_fn=pred_fn
        )
        accuracies_test, entropies_test = evaluate_on_all_tasks(
            self.test_iterators, pred_fn=pred_fn
        )
        return accuracies, entropies, accuracies_test, entropies_test


def _process_kwargs_protocol(kwargs: Dict) -> None:
    """Preprocess hyperparameters for the protocol."""
    kwargs["task"] = kwargs["data_training"]
    if kwargs["not_use_val_split"]:
        kwargs["use_val_split"] = False
    kwargs["kwargs"] = copy(kwargs)
