"""S-FSVI continual learning method."""
import copy
from functools import partial
from logging import Logger
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from jax import jit

from benchmarking.benchmark_args import NOT_SPECIFIED
from benchmarking.data_loaders.get_data import TUPLE_OF_TWO_TUPLES
from benchmarking.data_loaders.get_data import get_output_dim_fn
from benchmarking.method_cl_template import MethodCLTemplate
from sfsvi.fsvi_utils.context_points import make_context_points
from sfsvi.fsvi_utils.coreset.coreset import Coreset
from sfsvi.fsvi_utils.coreset.coreset_selection import get_coreset_indices
from sfsvi.fsvi_utils.coreset.coreset_selection import make_pred_fn
from sfsvi.fsvi_utils.initializer import Initializer
from sfsvi.fsvi_utils.replace_params import replace_opt_state_trained
from sfsvi.fsvi_utils.replace_params import replace_params_trained_heads
from sfsvi.fsvi_utils.utils_cl import get_minibatch
from sfsvi.fsvi_utils.utils_cl import initialize_random_keys
from sfsvi.general_utils.log import (
    Hyperparameters,
)
from sfsvi.models.networks import Model


class MethodCLFSVI(MethodCLTemplate):
    """S-FSVI continual learning method."""

    def __init__(
        self,
        logger: Logger,
        input_shape: List[int],
        n_train: int,
        range_dims_per_task: TUPLE_OF_TWO_TUPLES,
        output_dim: int,
        n_coreset_inputs_per_task_list: Tuple[int, ...],
        kwargs: Dict,
    ):
        """
        :param logger: a logger for logging results.
        :param input_shape: shape of input image including batch dimension,
            batch dimension is 1.
        :param n_train: number of training samples.
        :param range_dims_per_task: output heads index range for each task.
            For example, for split MNIST (MH), this variable is
                `range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
            which means output heads for the first task are the 1st and 2nd
            output dimensions, the output heads for the second task are the
            3rd and 4th dimension, etc.
        :param output_dim: the number of output dimensions.
        :param n_coreset_inputs_per_task_list: the number of maximum allowed
            coreset points for each task. A coreset is a small set of input
            points that can be stored and used when training on future tasks
            for helping avoid forgetting.
        :param kwargs: the hyperparameters of this continual learning method.
        """
        self.n_coreset_inputs_per_task_list = n_coreset_inputs_per_task_list
        self.range_dims_per_task = range_dims_per_task
        self.logger = logger
        self.input_shape = input_shape
        self.n_train = n_train
        self.output_dim = output_dim
        self._initialization(kwargs)
        self.params, self.params_prior = self.params_init, self.params_init
        self.state = self.state_init
        self.opt_state = self.opt.init(self.params_init)

    def run_one_task(
        self,
        task_id: int,
        train_dataset: tf.data.Dataset,
        n_train: int,
        validation_dataset: Optional[tf.data.Dataset],
        n_valid: Optional[int],
        fn_for_logging_info_per_epoch: Callable,
        logger: Logger,
    ) -> Dict:
        """Train model on one task and evaluate the model on tasks seen so far.

        Read the documentation of this method of the parent class.
        """
        batch_train_iterator = iter(train_dataset.batch(self.hparams.batch_size))
        full_train_iterator = iter(train_dataset.batch(n_train))
        # redefine model if self.hparams.only_trainable_head is true
        if task_id > 0 and self.hparams.only_trainable_head:
            self.params, self.opt_state = self.reset_output_dim(
                task_id=task_id, params_old=self.params, opt_state_old=self.opt_state
            )
            self.params_prior = copy.copy(self.params)

        print(f"\nLearning task {task_id + 1}")
        n_batches = n_train // self.hparams.batch_size
        nb_epochs = (
            int(self.hparams.epochs_first_task)
            if task_id == 0
            else int(self.hparams.epochs)
        )
        prior_fn = self._update_prior_per_task(
            params_prior=self.params_prior,
            task_id=task_id,
            state=self.state,
        )
        # redefine optimizer if we want to change learning rate
        if self.hparams.learning_rate_first_task != NOT_SPECIFIED and task_id == 0:
            learning_rate = float(self.hparams.learning_rate_first_task)
        else:
            learning_rate = float(self.hparams.learning_rate)
        self.opt = self.initializer.initialize_optimizer(learning_rate=learning_rate)

        for epoch in range(nb_epochs):
            self._run_one_epoch(
                task_id=task_id,
                epoch=epoch,
                n_batches=n_batches,
                train_iterator=batch_train_iterator,
                prior_fn=prior_fn,
                fn_logging_info_per_epoch=fn_for_logging_info_per_epoch,
            )

        self._add_points_to_coreset(
            full_train_iterator=full_train_iterator,
            task_id=task_id,
            params=self.params,
            state=self.state,
            params_prior=self.params_prior,
        )

        # Use the posterior from this task as the prior for the next task
        self.params_prior = copy.copy(self.params)
        info_to_save = {
            "coreset": self.coreset.wrap_data_to_save(latest_task=True),
        }
        if self.hparams.save_all_params:
            info_to_save.update(
                {
                    "params": self.params,
                    "state": self.state,
                    "opt_state": self.opt_state,
                }
            )
        return info_to_save

    def _run_one_epoch(
        self,
        task_id: int,
        epoch: int,
        n_batches: int,
        train_iterator: Iterator,
        prior_fn: Callable,
        fn_logging_info_per_epoch: Callable,
    ) -> None:
        """Train the model for one epoch and evaluate the model on tasks
        seen so far."""
        self.logger.info(f"task_id {task_id}, epoch {epoch} starts")
        for _ in range(n_batches):
            self.params, self.state, self.opt_state = self._run_one_optimization_step(
                params=self.params,
                state=self.state,
                train_iterator=train_iterator,
                task_id=task_id,
                opt_state=self.opt_state,
                prior_fn=prior_fn,
            )
        fn_logging_info_per_epoch(
            pred_fn=self.get_pred_fn(),
            epoch=epoch,
        )

    def _run_one_optimization_step(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state,
        train_iterator: Iterator,
        prior_fn: Callable,
        task_id: int,
    ) -> Tuple[hk.Params, hk.State, Any]:
        data = next(train_iterator)
        x_batch, y_batch = get_minibatch(
            data, self.output_dim, self.input_shape
        )

        if task_id == 0 and self.hparams.n_context_points_first_task != NOT_SPECIFIED:
            n_context_points = int(self.hparams.n_context_points_first_task)
        elif (
            task_id == 1 and self.hparams.n_context_points_second_task != NOT_SPECIFIED
        ):
            n_context_points = int(self.hparams.n_context_points_second_task)
        elif (
            self.hparams.coreset_n_tasks != NOT_SPECIFIED
            and task_id < int(self.hparams.coreset_n_tasks)
            and self.hparams.context_point_adjustment
        ):
            n_context_points = (
                int(self.hparams.n_context_point_adjust_amount) // task_id
            )
        else:
            n_context_points = self.hparams.n_context_points

        x_context = make_context_points(
            not_use_coreset=self.hparams.not_use_coreset,
            constant_context_points=self.hparams.constant_context_points,
            n_context_points=n_context_points,
            context_point_augmentation=self.hparams.context_point_augmentation,
            task_id=task_id,
            kh=self.kh,
            x_batch=x_batch,
            context_point_fn=self.context_point_fn,
            coreset=self.coreset,
            draw_per_class=self.hparams.coreset == "random_per_class",
            coreset_n_tasks=self.hparams.coreset_n_tasks,
            n_augment=self.hparams.n_augment,
            augment_mode=self.hparams.augment_mode,
        )

        params, state, opt_state, loss_info = update(
            self.loss,
            params,
            state,
            opt_state,
            prior_fn,
            x_batch,
            y_batch,
            x_context,
            self.range_dims_per_task,
            task_id,
            self.kh.next_key(),
            self.opt,
            self.hparams.n_samples,
            self.model,
        )
        return params, state, opt_state

    def get_pred_fn(self) -> Callable:
        """Returns a function for predicting the class probabilities for
        a given task.

        :return
            a function takes in two keyword arguments:
                x: `jnp.ndarray`, input images to the model.
                task_id: `int`, 0-based task identifier.
            and returns a `jnp.ndarray` which is the predicted probabilities
            of different classes.
        """
        return make_pred_fn(
            model=self.model,
            params=self.params,
            state=self.state,
            rng_key=self.kh.next_key(),
            n_samples_eval=self.hparams.n_samples_eval,
            range_dims_per_task=self.range_dims_per_task,
        )

    def get_final_data_to_log(self) -> Dict:
        """Return final states to be saved as part of the artifacts."""
        to_save = {
            "params": self.params,
            "state": self.state,
            "coreset": self.coreset.wrap_data_to_save(latest_task=False),
        }
        return to_save

    def _initialization(
        self,
        kwargs: Dict,
    ):
        """Initialise model."""
        self.hparams = Hyperparameters(**kwargs)
        self.kh = initialize_random_keys(seed=self.hparams.seed)
        if self.hparams.only_trainable_head:
            assert (
                "omniglot" in self.hparams.data_training.lower()
            ), f"not implemented for {self.hparams.data_training}"
            self.output_dim = self.range_dims_per_task[0][1]
            self.output_dim_fn = get_output_dim_fn(self.range_dims_per_task)

        initializer = Initializer(
            hparams=self.hparams,
            input_shape=self.input_shape,
            output_dim=self.output_dim,
            n_marginals=self.hparams.n_marginals,
            stochastic_linearization=self.hparams.stochastic_linearization,
        )
        self.initializer = initializer
        (
            self.model,
            _,
            self.apply_fn,
            self.state_init,
            self.params_init,
        ) = initializer.initialize_model(self.kh.next_key())
        assert (
            not self.hparams.only_trainable_head or not self.state_init
        ), "not implemented for non empty state yet"
        self.opt = initializer.initialize_optimizer()
        self.cl_prior = initializer.initialize_cl_prior()
        self.context_point_fn = initializer.initialize_context_points_fn()
        metrics = initializer.initialize_objective(model=self.model)
        self.loss = metrics.nelbo_fsvi_classification_multihead
        self.coreset = Coreset()

    def reset_output_dim(
        self,
        task_id: int,
        params_old: hk.Params,
        opt_state_old: Any,
    ) -> Tuple[hk.Params, Any]:
        self.output_dim = self.output_dim_fn(task_id)
        self.cl_prior, model_tuple = self.initializer.reset_output_dim(
            self.output_dim, self.kh.next_key()
        )
        self.model, _, self.apply_fn, state_init, params_init = model_tuple
        params_new = replace_params_trained_heads(
            params_old=params_old,
            params_new=params_init,
        )
        metrics = self.initializer.initialize_objective(model=self.model)
        self.loss = metrics.nelbo_fsvi_classification_multihead
        opt_state_new = self.opt.init(params_new)
        opt_state_replaced = replace_opt_state_trained(
            opt_state_old=opt_state_old,
            opt_state_new=opt_state_new,
        )
        return params_new, opt_state_replaced

    def _add_points_to_coreset(
        self,
        full_train_iterator: Iterator,
        task_id: int,
        params: hk.Params,
        state: hk.State,
        params_prior: hk.Params,
    ) -> None:
        """Select data point to add to coreset."""
        # Add some points from the current task to the coreset
        x_candidate, y_candidate = next(full_train_iterator)
        x_candidate, y_candidate = (
            np.array(x_candidate),
            np.array(y_candidate),
        )
        n_add = self.n_coreset_inputs_per_task_list[task_id]
        inds_add = get_coreset_indices(
            hparams=self.hparams,
            x_candidate=x_candidate,
            y_candidate=y_candidate,
            n_add=n_add,
            model=self.model,
            params=params,
            state=state,
            kh=self.kh,
            range_dims_per_task=self.range_dims_per_task,
            task_id=task_id,
            prior=self.cl_prior,
            apply_fn=self.apply_fn,
            params_prior=params_prior,
            loss=self.loss,
            stochastic_linearization=self.hparams.stochastic_linearization,
        )
        self.coreset.add_coreset_points(
            x_candidate=x_candidate,
            y_candidate=y_candidate,
            inds_add=inds_add,
        )

    def _update_prior_per_task(
        self,
        state: hk.State,
        params_prior: hk.Params,
        task_id: int,
    ) -> Callable:
        """Update function for generating prior function distribution with the
         latest model parameters.

        :return
            a function that takes in
                x_context: `jnp.ndarray`, a batch of context points
            and returns a 2-tuple:
                `jnp.ndarray`, mean of the prior Gaussian distribution
                `jnp.ndarray`, variance of the prior Gaussian distribution
        """
        prior_fn = self.cl_prior.make_prior_fn(
            apply_fn=self.apply_fn,
            state=state,
            params=params_prior,
            rng_key=self.kh.next_key(),
            task_id=task_id,
            identity_cov=self.hparams.identity_cov,
            jit_prior=True,
        )
        if task_id > 0 and self.hparams.identity_cov:
            print(f"at task {task_id}, turning on identity_cov")
            prior_fn = partial(prior_fn, identity_cov=True)
        return prior_fn


@partial(jit, static_argnums=(0, 4, 8, 9, 11, 12, 13))
def update(
    loss: Callable,
    params: hk.Params,
    state: hk.State,
    opt_state: Any,
    prior_fn: Callable,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    x_context: Dict[int, np.ndarray],
    range_dims_per_task: TUPLE_OF_TWO_TUPLES,
    task_id: int,
    rng_key: jnp.ndarray,
    opt: optax.GradientTransformation,
    n_samples: int,
    model: Model,
) -> Tuple[hk.Params, hk.State, Any, Dict]:
    """Do one step of gradient update.

    :param loss: loss function.
    :param params: model parameters.
    :param state: model state, e.g. for batch normalisation state.
    :param opt_state: optax optimiser state.
    :param prior_fn: function for computing prior function distribution.
    :param x_batch: a batch of input images.
    :param y_batch: a batch of labels.
    :param x_context: a batch of context points for calculating KL-divergence
        term in the ELBO objective.
    :param range_dims_per_task: output heads index range for each task.
            For example, for split MNIST (MH), this variable is
                `range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
            which means output heads for the first task are the 1st and 2nd
            output dimensions, the output heads for the second task are the
            3rd and 4th dimension, etc.
    :param task_id: 0-based task id.
    :param rng_key: JAX random seed.
    :param opt: optax optimiser.
    :param n_samples: number of Monte-Carlo samples for estimating the expected
        log likelihood under variational distribution.
    :param model: model containing the forward pass function.
    :return:
        updated model parameters.
        updated model state.
        updated optimiser state.
        debugging information during the forward pass, such as the different
            components of the ELBO objective (i.e. log likelihood, KL)
    """
    prior_means, prior_covs = prior_fn(context_points=x_context)
    grads, other_info = jax.grad(loss, argnums=0, has_aux=True)(
        params,
        state,
        prior_means,
        prior_covs,
        x_batch,
        y_batch,
        x_context,
        rng_key,
        range_dims_per_task,
        task_id,
        n_samples,
        True,  # is_training
    )
    # log_gpu_usage("gradient step")
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    state = update_state(
        model=model, params=params, state=state, rng_key=rng_key, x_batch=x_batch
    )
    return params, state, opt_state, other_info


def update_state(
    model: Model,
    params: hk.Params,
    state: hk.State,
    rng_key: jnp.ndarray,
    x_batch: jnp.ndarray,
) -> hk.State:
    """Update the model state (e.g. batch normalisation state) by performing
    one forward pass."""
    if len(state):
        state = model.apply_fn(
            params, state, rng_key, x_batch, rng_key, stochastic=True, is_training=True
        )[1]
    return state
