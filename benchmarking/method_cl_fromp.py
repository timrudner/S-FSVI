from logging import Logger
from typing import Callable
from typing import Dict

import jax
import numpy as np
import tensorflow as tf
import torch
from torch import nn

from baselines.fromp.models_fromp import MLP
from baselines.fromp.models_fromp import SplitMLP
from baselines.fromp.opt_fromp import opt_fromp
from baselines.fromp.utils_fromp import process_data
from baselines.fromp.utils_fromp import random_memorable_points
from baselines.fromp.utils_fromp import select_memorable_points
from baselines.fromp.utils_fromp import update_fisher
from benchmarking.method_cl_template import MethodCLTemplate
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class
from sfsvi.fsvi_utils.utils_cl import TUPLE_OF_TWO_TUPLES
from sfsvi.fsvi_utils.utils_cl import select_context_points
from sfsvi.general_utils.log import Hyperparameters


class MethodCLFROMP(MethodCLTemplate):
    def __init__(
        self,
        input_shape,
        output_dim,
        n_coreset_inputs_per_task_list,
        range_dims_per_task,
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
        :param range_dims_per_task: output heads index range for each task.
            For example, for split MNIST (MH), this variable is
                `range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
            which means output heads for the first task are the 1st and 2nd
            output dimensions, the output heads for the second task are the
            3rd and 4th dimension, etc.
        :param kwargs: the hyperparameters of this continual learning method.
        """
        self.hparams = Hyperparameters(**kwargs)
        self.use_cuda = torch.cuda.is_available()
        self.task = self.hparams.data_training
        lr = self.hparams.lr
        tau = self.hparams.tau
        self.range_dims_per_task = tuple(
            [(d0, d1 - 1) for d0, d1 in range_dims_per_task]
        )
        self.n_context_points_per_task = n_coreset_inputs_per_task_list

        hidden_layers = (self.hparams.hidden_size,) * self.hparams.n_layers
        layer_size = (np.prod(input_shape),) + hidden_layers + (output_dim,)
        if is_multihead(self.task):
            model = SplitMLP(layer_size, act="relu")
        elif is_single_head(self.task):
            model = MLP(layer_size, act="relu")
        else:
            raise ValueError(f"haven't configured for self.task {self.task}")

        criterion = torch.nn.CrossEntropyLoss()
        if self.use_cuda:
            criterion.cuda()
            model.cuda()
        self.criterion = criterion

        if "smnist" in self.task:
            optimizer = opt_fromp(
                model, lr=lr, prior_prec=1e-3, grad_clip_norm=0.1, tau=tau
            )
        elif "pmnist" in self.task:
            optimizer = opt_fromp(
                model, lr=lr, prior_prec=1e-5, grad_clip_norm=0.01, tau=tau
            )
        elif "sfashionmnist" in self.task:
            optimizer = opt_fromp(
                model, lr=lr, prior_prec=1e-3, grad_clip_norm=0.1, tau=tau
            )
        elif "smnist_sh" in self.task:
            optimizer = opt_fromp(
                model, lr=lr, prior_prec=1e-5, grad_clip_norm=0.01, tau=tau
            )
        else:
            raise ValueError(f"haven't configured optimizer for self.task {self.task}")
        self.model = model
        self.optimizer = optimizer

        self.memorable_points = []

        np.random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

    def run_one_task(
        self,
        task_id: int,
        train_dataset: tf.data.Dataset,
        n_train: int,
        validation_dataset: tf.data.Dataset,
        n_valid: int,
        fn_for_logging_info_per_epoch: Callable[[Callable, int], None],
        logger: Logger,
    ):
        """Train model on one task and evaluate the model on tasks seen so far.

        Read the documentation of this method of the parent class.
        """
        print(f"start working on task {task_id}")
        if task_id > 0:

            def closure(task_id):
                # Calculate and store regularisation-term-related quantities
                self.optimizer.zero_grad()
                memorable_points_t = self.memorable_points[task_id][0]
                if self.use_cuda:
                    memorable_points_t = memorable_points_t.cuda()
                logits = get_logits(
                    model=self.model,
                    task=self.task,
                    range_dims_per_task=self.range_dims_per_task,
                    task_id=task_id,
                    x=memorable_points_t,
                )
                return logits

            if is_smnist_multihead(self.task):
                self.optimizer.init_task(closure, task_id, eps=self.hparams.smnist_eps)
            elif "pmnist" in self.task:
                self.optimizer.init_task(closure, task_id, eps=self.hparams.smnist_eps)
            elif "sfashionmnist" in self.task:
                self.optimizer.init_task(closure, task_id, eps=self.hparams.smnist_eps)
            elif "smnist_sh" in self.task:
                self.optimizer.init_task(closure, task_id, eps=self.hparams.smnist_eps)
            else:
                raise ValueError(f"haven't configured for task {self.task}")

        n_batches = n_train // self.hparams.batch_size

        self.model.train()
        batch_train_iterator = iter(train_dataset.batch(self.hparams.batch_size))
        full_train_iterator = iter(train_dataset.batch(n_train))

        for epoch in range(self.hparams.n_epochs):
            for batch_id in range(n_batches):
                x, y = next(batch_train_iterator)
                x, y = process_data(x, y, self.range_dims_per_task[task_id])
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()

                def closure():
                    # Closure on current task's data
                    self.optimizer.zero_grad()
                    logits = get_logits(
                        model=self.model,
                        task=self.task,
                        range_dims_per_task=self.range_dims_per_task,
                        task_id=task_id,
                        x=x,
                    )
                    loss = self.criterion(logits, y)
                    return loss, logits

                def closure_memorable_points(task_id):
                    # Closure on memorable past data
                    self.optimizer.zero_grad()
                    memorable_points_t = self.memorable_points[task_id][0]
                    if self.use_cuda:
                        memorable_points_t = memorable_points_t.cuda()
                    logits = get_logits(
                        model=self.model,
                        task=self.task,
                        range_dims_per_task=self.range_dims_per_task,
                        task_id=task_id,
                        x=memorable_points_t,
                    )
                    return logits

                loss, logits = self.optimizer.step(
                    closure, closure_memorable_points, task_id
                )
                if self.hparams.n_steps != "not_specified" and batch_id + 1 == int(
                    self.hparams.n_steps
                ):
                    print("break out in advance")
                    break

        self.model.eval()

        def _pred_fn(task_id, x):
            x, _ = process_data(x, y=None, range_dims=self.range_dims_per_task[task_id])
            if self.use_cuda:
                x = x.cuda()
            logits = get_logits(
                model=self.model,
                task=self.task,
                range_dims_per_task=self.range_dims_per_task,
                task_id=task_id,
                x=x,
            )
            if self.use_cuda:
                logits = logits.cpu()
            return logits.detach().numpy()

        fn_for_logging_info_per_epoch(pred_fn=_pred_fn, epoch=self.hparams.n_epochs)

        if is_multihead(self.task):
            label_set = self.range_dims_per_task[task_id]
        elif is_single_head(self.task):
            label_set = None

        if "lambda" in self.hparams.select_method:
            n_classes = (
                self.range_dims_per_task[task_id][1]
                - self.range_dims_per_task[task_id][0]
                + 1
            )
            print("nb classes", n_classes)
            memorable_points_task = select_memorable_points(
                batch_train_iterator,
                self.model,
                n_batches,
                n_points=self.n_context_points_per_task[task_id],
                n_classes=n_classes,
                use_cuda=self.use_cuda,
                label_set=label_set,
                descending=("descend" in self.hparams.select_method),
            )
        elif self.hparams.select_method == "random_choice":
            memorable_points_task = random_memorable_points(
                batch_train_iterator,
                n_points=self.n_context_points_per_task[task_id],
                n_classes=len(self.range_dims_per_task[task_id]),
            )
        elif self.hparams.select_method == "random_noise":
            memorable_points_task = [
                torch.rand(self.n_context_points_per_task[task_id], 784),
            ]
        elif "train_pixel_rand" in self.hparams.select_method:
            x, y = next(full_train_iterator)
            memorable_points_task_array = select_context_points(
                n_context_points=self.n_context_points_per_task[task_id],
                context_point_type=self.hparams.select_method,
                context_points_bound=None,
                input_shape=[1, 784],
                x_batch=x.numpy(),
                rng_key=jax.random.PRNGKey(self.hparams.seed),
            )
            memorable_points_task = [torch.from_numpy(memorable_points_task_array)]
        elif self.hparams.select_method == "random_per_class":
            x, y = next(batch_train_iterator)
            x_candidate, y_candidate = x.numpy(), y.numpy()
            inds_add = add_by_random_per_class(
                y_candidate=y_candidate, n_add=self.n_context_points_per_task[task_id]
            )
            print("inducing points")
            print(y_candidate[inds_add])
            memorable_points_task_array = x_candidate[inds_add]
            memorable_points_task = [torch.from_numpy(memorable_points_task_array)]
        else:
            raise ValueError(
                f"Invalid value for self.hparams.select_method: {self.hparams.select_method}"
            )

        print("memorable points appended!")
        self.memorable_points.append(memorable_points_task)

        update_fisher(
            batch_train_iterator,
            n_batches,
            self.model,
            self.optimizer,
            use_cuda=self.use_cuda,
            label_set=label_set,
        )
        print("updated fisher!")
        return {}


def get_logits(
    model: nn.Module,
    task: str,
    range_dims_per_task: TUPLE_OF_TWO_TUPLES,
    task_id: int,
    x: np.ndarray,
) -> np.ndarray:
    """Compute logits for a given task."""
    if is_multihead(task):
        range_dims = range_dims_per_task[task_id]
        logits = model.forward(x, range_dims)
    elif is_single_head(task):
        logits = model.forward(x)
    else:
        raise ValueError(f"haven't configured for task {task}")
    return logits


def is_multihead(task: str) -> bool:
    return is_smnist_multihead(task) or "sfashionmnist" in task


def is_smnist_multihead(task: str) -> bool:
    return "smnist" in task and "smnist_sh" not in task


def is_single_head(task: str) -> bool:
    return "pmnist" in task or "pfashionmnist" in task or "smnist_sh" in task
