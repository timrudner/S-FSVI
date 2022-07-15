from functools import partial
from logging import Logger
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import List
from typing import Tuple

import gpflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sonnet as snt

from baselines.frcl.utils_frcl import ContinualGPmodel
from baselines.frcl.utils_frcl import ConvNetworkWithBias
from baselines.frcl.utils_frcl import MLPNetworkWithBias
from baselines.frcl.utils_frcl import predict_at_head_frcl
from benchmarking.method_cl_template import MethodCLTemplate
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class
from sfsvi.general_utils.log import Hyperparameters


class MethodCLFRCL(MethodCLTemplate):
    def __init__(
        self,
        input_shape: List[int],
        output_dim: int,
        n_coreset_inputs_per_task_list: Tuple[int, ...],
        kwargs: Dict,
    ):
        """
        :param input_shape: shape of input image including batch dimension,
            batch dimension is 1.
        :param output_dim: the number of output dimensions.
        :param n_coreset_inputs_per_task_list: the number of maximum allowed
            coreset points for each task. A coreset is a small set of input
            points that can be stored and used when training on future tasks
            for helping avoid forgetting.
        :param kwargs: the hyperparameters of this continual learning method.
        """
        self.hparams = Hyperparameters(**kwargs)
        hidden_sizes = [self.hparams.hidden_size] * self.hparams.n_layers
        base_network = compose_base_network(
            task=self.hparams.data_training, hidden_sizes=hidden_sizes
        )
        self.FRCL = ContinualGPmodel(
            num_features=(hidden_sizes[-1] + 1),
            num_classes=output_dim,
            base_network=base_network,
            likelihood=gpflow.likelihoods.MultiClass(output_dim),
        )

        self.all_accuracies, self.all_iterators = [], []
        self.select_method = self.hparams.select_method
        self.seed = self.hparams.seed
        self.learning_rate = self.hparams.learning_rate
        self.n_iterations_train = self.hparams.n_iterations_train
        self.n_coreset_inputs_per_task_list = n_coreset_inputs_per_task_list
        self.n_iterations_discr_search = self.hparams.n_iterations_discr_search
        self.input_shape = input_shape

    def run_one_task(
        self,
        task_id: int,
        train_dataset: tf.data.Dataset,
        n_train: int,
        validation_dataset: tf.data.Dataset,
        n_valid: int,
        fn_for_logging_info_per_epoch: Callable[[Callable, int], None],
        logger: Logger,
    ) -> Dict:
        """Train model on one task and evaluate the model on tasks seen so far.

        Read the documentation of this method of the parent class.
        """
        rng_state = np.random.RandomState(self.seed)
        print(f"Learning task {task_id+1}")
        if logger:
            logger.info(f"Learning task {task_id+1}")

        self.FRCL.get_weight_space_approx(
            rng_state
        )  # weight-space approximation for the current task
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate
        )  # task-specific optimiser

        batch_train_iterator = iter(train_dataset.batch(self.hparams.batch_size))
        full_train_iterator = iter(train_dataset.batch(n_train))
        for _ in tqdm(range(self.n_iterations_train)):
            x, y = next(batch_train_iterator)
            loss = lambda: self.FRCL.objective_weight_space(x, y, n_train)[0]
            optimizer.minimize(loss)

        # Randomly select inducing points from the discrete-search set
        x_discr, y_discr = next(full_train_iterator)
        permutation_train = np.random.permutation(n_train)
        n_inputs_to_add = self.n_coreset_inputs_per_task_list[task_id]
        inds_inducing = permutation_train[-n_inputs_to_add:]
        x_inducing = tf.gather(x_discr, inds_inducing, axis=0)

        full_valid_iterator = iter(validation_dataset.batch(n_valid))
        if self.select_method == "trace_term":
            x_discr_eval, _ = next(
                full_valid_iterator
            )  # data to evaluate the inducing set on
            loss_current = self.FRCL.trace_term(x_discr_eval, x_inducing)
            results_discr_search = [(0, loss_current, x_inducing)]
            train_set_id = (
                0  # candidate training point to replace the inducing point with
            )
            n_accepted_moves = 0

            for iteration in range(self.n_iterations_discr_search):
                inducing_set_id = iteration % n_inputs_to_add
                inds_inducing_proposed = inds_inducing.copy()

                # Replace inducing point and re-evaluate
                inds_inducing_proposed[inducing_set_id] = permutation_train[
                    train_set_id
                ]
                x_inducing_proposed = tf.gather(x_discr, inds_inducing_proposed, axis=0)
                loss_proposed = self.FRCL.trace_term(x_discr_eval, x_inducing_proposed)

                if loss_proposed < loss_current:
                    inds_inducing = inds_inducing_proposed  # FIX 3: update `inds_inducing` instead of copying
                    x_inducing = x_inducing_proposed  # FIX 1: update `x_inducing`
                    loss_current = loss_proposed
                    results_discr_search.append((iteration, loss_current, x_inducing))
                    n_accepted_moves += 1

                if train_set_id == n_train - 1:
                    permutation_train = np.random.permutation(n_train)
                    train_set_id = 0
                else:
                    train_set_id += 1

            print(
                f"Discrete search for inducing points: {n_accepted_moves} accepted moves"
            )
            if logger:
                logger.info(
                    f"Discrete search for inducing points: {n_accepted_moves} accepted moves"
                )
            _, _, x_inducing_init = results_discr_search[0]

        elif self.select_method == "random_choice":
            x_inducing_init = x_inducing
        elif self.select_method == "random_noise":
            x_inducing = tf.random.uniform(
                (n_inputs_to_add,) + tuple(self.input_shape[1:]), dtype=tf.float64
            )
            x_inducing_init = x_inducing
        elif self.select_method == "random_per_class":
            x_candidate, y_candidate = x_discr.numpy(), y_discr.numpy()
            inds_add = add_by_random_per_class(
                y_candidate=y_candidate, n_add=n_inputs_to_add
            )
            print("inducing points")
            print(y_candidate[inds_add])
            if logger:
                logger.info("inducing points")
                logger.info(y_candidate[inds_add])
            x_inducing = tf.convert_to_tensor(x_candidate[inds_add])
            x_inducing_init = x_inducing
        else:
            raise ValueError("Invalid value for select_method")

        self.FRCL.complete_task_weight_space(x_inducing, x_inducing_init)
        pred_fn = partial(predict_at_head_frcl, model=self.FRCL)
        # TODO: remove "epoch" from the interface as it is not generic enough
        fn_for_logging_info_per_epoch(pred_fn=pred_fn, epoch=self.n_iterations_train)
        return {}


def compose_base_network(task: str, hidden_sizes: Sequence[int]) -> snt.Module:
    if "omniglot" in task:
        recommended_hidden_sizes = [64, 64, 64, 64]
        if hidden_sizes != recommended_hidden_sizes:
            print(
                f"Not using default hidden sizes in FRCL: "
                f"default = {recommended_hidden_sizes}, "
                f"hidden_sizes = {hidden_sizes}"
            )
        base_network = ConvNetworkWithBias(output_sizes=hidden_sizes)
    else:
        base_network = MLPNetworkWithBias(output_sizes=hidden_sizes)
    return base_network
