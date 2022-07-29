import copy
from functools import partial
from typing import Callable
from typing import Dict

import haiku as hk
import jax
import numpy as np
import optax
import tensorflow as tf
from jax import jit

from benchmarking.method_cl_template import MethodCLTemplate
from sfsvi.general_utils.log import (
    Hyperparameters,
)
from sfsvi.fsvi_utils.utils_cl import get_minibatch
from sfsvi.fsvi_utils.utils_cl import initialize_random_keys
from sfsvi.fsvi_utils.args_cl import NOT_SPECIFIED
from sfsvi.fsvi_utils.coreset.coreset import Coreset
from sfsvi.fsvi_utils.coreset.coreset_selection import get_coreset_indices
from sfsvi.fsvi_utils.coreset.coreset_selection import make_pred_fn
from benchmarking.data_loaders.get_data import TUPLE_OF_TWO_TUPLES
from benchmarking.data_loaders.get_data import get_output_dim_fn
from sfsvi.fsvi_utils.inducing_points import make_inducing_points
from sfsvi.fsvi_utils.initializer import Initializer
from sfsvi.fsvi_utils.replace_params import replace_opt_state_trained
from sfsvi.fsvi_utils.replace_params import replace_params_trained_heads
from sfsvi.fsvi_utils.utils_cl import TrainingLog


class MethodCLFSVI(MethodCLTemplate):
    """
    Rule of assigning an object to be an attribute of the class: only if the attribute always points to the same
    object.

    Counterexample: a new model parameter instance is created at every optimization step, so it can not be
        an attribute.
    """

    def __init__(
        self,
        logger,
        input_shape,
        n_train,
        range_dims_per_task,
        output_dim: int,
        n_coreset_inputs_per_task_list,
        method_kwargs: Dict,
    ):
        self.n_coreset_inputs_per_task_list = n_coreset_inputs_per_task_list
        self.range_dims_per_task = range_dims_per_task
        self.logger = logger
        self.input_shape = input_shape
        self.n_train = n_train
        self.output_dim = output_dim
        self._initialization(method_kwargs)
        # initialize internal states
        self.params, self.params_prior = self.params_init, self.params_init
        self.state = self.state_init
        self.opt_state = self.opt.init(self.params_init)

    def run_one_task(
        self,
        task_id: int,
        train_dataset: tf.data.Dataset,
        n_train: int,
        fn_for_logging_info_per_epoch: Callable,
    ):
        """
        :param task_id: (0-indexed) task number, for example, the second task
            has a task_id of 1.
        # TODO: should only receive training data.
        :param task_data:
        :param fn_for_logging_info_per_epoch: a Callable for evaluating
            the model at the end of each epoch.
        :return:
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
            params_prior=self.params_prior, task_id=task_id, state=self.state,
        )
        # redefine optimizer if we want to change learning rate
        if self.hparams.learning_rate_first_task != NOT_SPECIFIED and task_id == 0:
            learning_rate = float(self.hparams.learning_rate_first_task)
        else:
            learning_rate = float(self.hparams.learning_rate)
        self.opt = self.initializer.initialize_optimizer(
            learning_rate=learning_rate
        )

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
            batch_train_iterator=batch_train_iterator,
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
            info_to_save.update({
                "params": self.params,
                "state": self.state,
                "opt_state": self.opt_state
            })
        return info_to_save

    def _run_one_epoch(
        self,
        task_id,
        epoch,
        n_batches,
        train_iterator,
        prior_fn,
        fn_logging_info_per_epoch: Callable,
    ):
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
            pred_fn=self.get_pred_fn(), epoch=epoch,
        )

    def _run_one_optimization_step(
        self,
        params,
        state,
        opt_state,
        train_iterator,
        prior_fn,
        task_id,
    ):
        data = next(train_iterator)
        x_batch, y_batch = get_minibatch(
            data, self.output_dim, self.input_shape, self.prediction_type
        )

        if task_id == 0 and self.hparams.n_inducing_inputs_first_task != NOT_SPECIFIED:
            n_inducing_inputs = int(self.hparams.n_inducing_inputs_first_task)
        elif (
            task_id == 1 and self.hparams.n_inducing_inputs_second_task != NOT_SPECIFIED
        ):
            n_inducing_inputs = int(self.hparams.n_inducing_inputs_second_task)
        elif (
            self.hparams.coreset_n_tasks != NOT_SPECIFIED
            and task_id < int(self.hparams.coreset_n_tasks)
            and self.hparams.inducing_input_adjustment
        ):
            n_inducing_inputs = (
                int(self.hparams.n_inducing_input_adjust_amount) // task_id
            )
        else:
            n_inducing_inputs = self.hparams.n_inducing_inputs

        x_inducing = make_inducing_points(
            not_use_coreset=self.hparams.not_use_coreset,
            constant_inducing_points=self.hparams.constant_inducing_points,
            n_inducing_inputs=n_inducing_inputs,
            inducing_input_augmentation=self.hparams.inducing_input_augmentation,
            task_id=task_id,
            kh=self.kh,
            x_batch=x_batch,
            inducing_input_fn=self.inducing_input_fn,
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
            x_inducing,
            self.range_dims_per_task,
            task_id,
            self.kh.next_key(),
            self.opt,
            self.hparams.n_samples,
            self.model,
            self.hparams.loss_type,
        )
        return params, state, opt_state

    def get_pred_fn(self):
        return make_pred_fn(
            model=self.model,
            params=self.params,
            state=self.state,
            rng_key=self.kh.next_key(),
            n_samples_eval=self.hparams.n_samples_eval,
            range_dims_per_task=self.range_dims_per_task,
        )

    def get_final_data_to_log(self) -> Dict:
        to_save = {
            "params": self.params,
            "state": self.state,
            "coreset": self.coreset.wrap_data_to_save(latest_task=False),
        }
        return to_save

    def _initialization(
        self, method_kwargs: Dict,
    ):
        self.hparams = Hyperparameters(**method_kwargs)
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
            n_train=self.n_train,
            n_batches=self.n_train // self.hparams.batch_size,
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
        self.prediction_type = "classification"
        self.cl_prior = initializer.initialize_cl_prior()
        self.inducing_input_fn = initializer.initialize_inducing_input_fn()
        metrics = initializer.initialize_objective(model=self.model)
        self.loss = metrics.nelbo_fsviassification_multihead
        self.coreset = Coreset()

    def reset_output_dim(self, task_id, params_old: hk.Params, opt_state_old):
        self.output_dim = self.output_dim_fn(task_id)
        self.cl_prior, model_tuple = self.initializer.reset_output_dim(
            self.output_dim, self.kh.next_key()
        )
        self.model, _, self.apply_fn, state_init, params_init = model_tuple
        params_new = replace_params_trained_heads(
            params_old=params_old, params_new=params_init,
        )
        metrics = self.initializer.initialize_objective(model=self.model)
        self.loss = metrics.nelbo_fsviassification_multihead
        opt_state_new = self.opt.init(params_new)
        opt_state_replaced = replace_opt_state_trained(
            opt_state_old=opt_state_old, opt_state_new=opt_state_new,
        )
        return params_new, opt_state_replaced

    def _add_points_to_coreset(
        self, batch_train_iterator, full_train_iterator, task_id, params, state, params_prior
    ):
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
            x_candidate=x_candidate, y_candidate=y_candidate, inds_add=inds_add,
        )

    def _update_prior_per_task(self, state, params_prior, task_id):
        # update prior_fn using the latest model parameters (through params_prior)
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


@partial(jit, static_argnums=(0, 4, 8, 9, 11, 12, 13, 14))
def update(
    loss,
    params,
    state,
    opt_state,
    prior_fn,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    x_inducing: Dict[int, np.ndarray],
    range_dims_per_task: TUPLE_OF_TWO_TUPLES,
    task_id,
    rng_key,
    opt,
    n_samples,
    model,
    loss_type,
):
    prior_means, prior_covs = prior_fn(inducing_inputs=x_inducing)
    grads, other_info = jax.grad(loss, argnums=0, has_aux=True)(
        params,
        state,
        prior_means,
        prior_covs,
        x_batch,
        y_batch,
        x_inducing,
        rng_key,
        range_dims_per_task,
        task_id,
        n_samples,
        True,  # is_training
        loss_type,
    )
    # log_gpu_usage("gradient step")
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    state = update_state(
        model=model, params=params, state=state, rng_key=rng_key, x_batch=x_batch
    )
    return params, state, opt_state, other_info


def update_state(model, params, state, rng_key, x_batch):
    if len(state):
        state = model.apply_fn(
            params, state, rng_key, x_batch, rng_key, stochastic=True, is_training=True
        )[1]
    return state
