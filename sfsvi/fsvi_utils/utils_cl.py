import os
import random as random_py
from copy import copy
from functools import partial
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import haiku as hk
import jax
import numpy as np
import pandas as pd
import tabulate
import tensorflow as tf
import torch
import tree
from jax import jit
from jax import numpy as jnp
from jax import random
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from sfsvi.models.haiku_mod import KeyHelper
from sfsvi.general_utils.log import save_chkpt

eps = 1e-12
kl_logstd_jitter = 1e-10
kl_cov_jitter = 1e-3
TWO_TUPLE = Tuple[int, int]
TUPLE_OF_TWO_TUPLES = Tuple[TWO_TUPLE, ...]


def predict_at_head_fsvi(
	x: jnp.ndarray,
	task_id: int,
	pred_fn: Callable,
	range_dims_per_task: TUPLE_OF_TWO_TUPLES,
	eval_batch_size: int = 1000,
) -> jnp.ndarray:
	"""Returns predicted class probabilties for the current task and set
	output heads for the other tasks to be 0.

	Make predictions then zero out the dimensions not corresponding to the
	output head for this task, so that when we pick the class with maximum
	predicted probability, only head for this task can be picked.

	:param x: input array.
	:param task_id: task id of the current task.
	:param pred_fn: function for making prediction, it takes a batch of
			array, and returns a 3-tuple containing the logit samples of shape
			[S, B, L], mean logit of shape [B, L], variance of logit of shape [B, L]
			where S is the number of samples, B is batch size, L is the number of
			logits.
	:param range_dims_per_task: output heads index range for each task.
			For example, for split MNIST (MH), this variable is
											`range_dims_per_task = [(0, 2), (2, 4), (4, 8), (8, 10)]`
			which means output heads for the first task are the 1st and 2nd
			output dimensions, the output heads for the second task are the
			3rd and 4th dimension, etc.
	:param eval_batch_size: batch size during evaluation.
	:return
									predicted class probabilties, an array of shape
																	(x.shape[0], output_dimension)
	"""
	pred_mean = _make_prediction_by_batch(
		x=x, pred_fn=pred_fn, eval_batch_size=eval_batch_size
	)
	preds = np.zeros_like(pred_mean)
	min_dim, max_dim = range_dims_per_task[task_id]
	preds[:, min_dim:max_dim] = jax.nn.softmax(pred_mean[:, min_dim:max_dim])
	return preds


def _make_prediction_by_batch(
	x: jnp.ndarray, pred_fn: Callable, eval_batch_size: int = 1000
) -> jnp.ndarray:
	"""Returns logits of prediction function on input array `x`.

	:param x: input array.
	:param pred_fn: function for making prediction, it takes a batch of
			array, and returns a 3-tuple containing the logit samples of shape
			[S, B, L], mean logit of shape [B, L], variance of logit of shape [B, L]
			where S is the number of samples, B is batch size, L is the number of
			logits.
	:param eval_batch_size: batch size during evaluation.
	:return:
	"""
	n_batches = int(np.ceil(x.shape[0] / eval_batch_size))
	preds_list = []
	for i in range(n_batches):
		_, _pred_mean, _ = pred_fn(
			inputs=x[i * eval_batch_size: (i + 1) * eval_batch_size]
		)
		preds_list.append(_pred_mean)
	pred_mean = np.concatenate(preds_list, axis=0)
	return pred_mean


def evaluate_on_all_tasks(
	iterators: Dict[int, Iterator], pred_fn: Callable
) -> Tuple[List[float], List[float]]:
	"""Evaluate prediction function on tasks seen so far.

	:param iterators: a list of data iterator (validation data), one for each
		task
	:param pred_fn: a function that takes in input and task_id, returns the
		model prediction (after softmax)
	:return:
		accuracies: a list of the same length as iterators, classification accuracy
		entropies: a list of the same length as iterators, classification entroopy
	"""
	accuracies, entropies = [], []
	for task_id, iterator in iterators.items():
		x, y = next(iterator)  # whole evaluation dataset for this task
		x, y = np.array(x), np.array(y)
		# preds is an array of shape (x.shape[0], task_dimension)
		preds = pred_fn(x=x, task_id=task_id) + eps
		assert y.ndim == 1, (
			"y is supposed to have only one dimension," f"y.shape = {y.shape}"
		)
		accuracy = np.mean(np.argmax(preds, axis=1) == y)
		entropy = np.mean(-np.sum((preds * np.log(preds)), axis=1))
		accuracies.append(round(accuracy, 4))
		entropies.append(round(entropy, 4))
	return accuracies, entropies


class TrainingLog:
	"""This class is responsible for collecting and displaying metrics."""

	def __init__(
		self,
		task_id: int,
	):
		"""
		:param task_id: index of task starting from zero
		"""
		self.task_id = task_id
		self.records = []

	def update(
		self,
		epoch: int,
		accuracies_test: List[float],
		**kwargs,
	) -> None:
		self.records.append(
			{"epoch": epoch, "accuracies_test": accuracies_test, **kwargs}
		)

	def print_progress(self, print_header: bool = False) -> None:
		"""Plot the epoch, ELBO, log likelihood, KL, accuracy.

		:param print_header: if True, print the header of table.
		"""
		record = self.records[-1]
		columns, values = ["epoch"], [record["epoch"]]
		columns.append("mean acc")
		values.append(f"{np.mean(record['accuracies_test']) * 100:.2f}")
		for i, acc in enumerate(record["accuracies_test"]):
			columns.append(f"t{i + 1} acc")
			values.append(f"{acc * 100:.2f}")

		table = tabulate.tabulate([values], columns, tablefmt="simple",
								  floatfmt="8.2f")
		if print_header:
			print("Nomenclature:\n\tacc: accuracy in %\n\tt1: the first task\n")
			table = table.split("\n")
			table = "\n".join([table[1]] + table)
		else:
			table = table.split("\n")[2]
		print(table)

	def print_short(self):
		"""
		Print average accuracy across tasks, and accuracy of each task.
		"""
		if self.records:
			test_accuracies = self.records[-1]["accuracies_test"]
			print(
				"For tasks seen so far,",
				f"\n---\nMean accuracy (test): {np.mean(test_accuracies):.4f}",
				f"\nAccuracies (test): {test_accuracies}\n---\n",
			)

	def get_dataframe(self, n_tasks: int) -> pd.DataFrame:
		"""Returns a dataframe containing the test accuracies of each tasks
		during the course of learning the task sequence.

		:param n_tasks: the total number of tasks
		:return:
				a pandas dataframe, each column corresponds to one task, each row
				corresponds to one log entry
		"""
		n_log_entries = len(self.records)
		accuracies = np.full(shape=(n_log_entries, n_tasks), fill_value=np.nan)
		for i, rec in enumerate(self.records):
			accuracies[i, : len(rec["accuracies_test"])] = rec[
				"accuracies_test"]
		data = {"task_id": self.task_id + 1}
		for task_id in range(n_tasks):
			data[f"test_acc_task{task_id + 1}"] = accuracies[:, task_id]
		df = pd.DataFrame(data)
		epochs = pd.DataFrame({"epoch": [r["epoch"] for r in self.records]})
		return pd.concat([epochs, df], axis=1)

	def save_task_specific_log(
		self,
		save_path: str,
		n_tasks: int,
		**kwargs,
	) -> None:
		"""Save task specific log.

		:param save_path: path where to save task-specific log.
		:param n_tasks: number of tasks seen so far.
		:param kwargs: other data to save.
		:return:
		"""
		current_task_data = {
			"training_log_dataframe": self.get_dataframe(n_tasks),
			**self.__dict__,
			**kwargs,
		}
		save_chkpt(p=save_path, **current_task_data)


def generate_4d_identity_cov(a: int, b: int) -> jnp.ndarray:
	"""Generate a 4-dimension identity covariance tensor.

	:param a: the first dimension.
	:param b: the second dimension.
	:return:
					an array of shape (a, b, a, b), where any slice of [:, i, :, i] or
					[j, :, j, :] is an identity matrix.
	"""
	return jnp.stack(
		jnp.stack(jnp.eye(a * b).split(a, 0)).split(a, 2)).transpose(
		(0, 2, 1, 3)
	)


def initialize_random_keys(seed: int) -> KeyHelper:
	"""Set random seed."""
	os.environ["PYTHONHASHSEED"] = str(seed)
	rng_key = jax.random.PRNGKey(seed)
	kh = KeyHelper(key=rng_key)
	random_py.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	torch.random.manual_seed(seed)
	return kh


def _one_hot(x: np.ndarray, k: int, dtype=jnp.float32) -> np.ndarray:
	"""Create a one-hot encoding of x of size k."""
	return np.array(x[:, None] == np.arange(k), dtype)


def get_minibatch(
	data: Tuple, output_dim: int, input_shape: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
	"""Returns a minibatch of data.

	:param data: a pair of (input, label).
	:param output_dim: the number of output dimensions.
	:param input_shape: the shape of input images.
	:return:
			input array.
			2D array of shape (batch_dim, output_dim).
	"""
	x, y = data
	if len(input_shape) <= 2:
		x_batch = np.reshape(x, [x.shape[0], -1])
	elif len(input_shape) == 4 and len(
		x.shape) != 4:  # handles flattened image inputs
		x_batch = np.array(x).reshape(
			x.shape[0], input_shape[1], input_shape[2], input_shape[3]
		)
	else:
		if x.shape[1] != x.shape[2]:
			x_batch = np.array(x).transpose([0, 2, 3, 1])
		else:
			x_batch = np.array(x)

	assert len(y.shape) == 1, "the label is supposed to have only one dimension"
	y_batch = _one_hot(np.array(y), output_dim)

	return x_batch, y_batch


@jit
def sigma_transform(params_log_var: hk.Params) -> hk.Params:
	"""Taking exponential of all arrays."""
	return tree.map_structure(lambda p: jnp.exp(p), params_log_var)


@partial(jit, static_argnums=(4,))
def kl_diag(
	mean_q: jnp.ndarray,
	mean_p: jnp.ndarray,
	cov_q: jnp.ndarray,
	cov_p: jnp.ndarray,
) -> jnp.ndarray:
	"""Return KL(q || p).

	All inputs are 1-D arrays.

	:param mean_q: mean of Gaussian distribution q.
	:param mean_p: mean of Gaussian distribution p.
	:param cov_q: covariance of Gaussian distribution q.
	:param cov_q: covariance of Gaussian distribution p.
	:return:
		KL divergence.
	"""
	kl_1 = jnp.log((cov_p + kl_logstd_jitter) ** 0.5) - jnp.log(
		(cov_q + kl_logstd_jitter) ** 0.5
	)
	kl_2 = ((cov_q + kl_logstd_jitter) + (mean_q - mean_p) ** 2) / (
		2 * (cov_p + kl_logstd_jitter)
	)
	kl_3 = -1 / 2
	kl = kl_1 + kl_2 + kl_3
	kl = jnp.sum(kl)
	return kl


@jit
def kl_full_cov(
	mean_q: jnp.ndarray,
	mean_p: jnp.ndarray,
	cov_q: jnp.ndarray,
	cov_p: jnp.ndarray,
) -> jnp.ndarray:
	"""Return KL(q || p).

	:param mean_q: mean of Gaussian distribution q.
	:param mean_p: mean of Gaussian distribution p.
	:param cov_q: covariance of Gaussian distribution q, 2-D array.
	:param cov_q: covariance of Gaussian distribution p, 2-D array.
	:return:
		KL divergence.
	"""
	dims = mean_q.shape[0]
	_cov_q = cov_q + jnp.eye(dims) * kl_cov_jitter
	_cov_p = cov_p + jnp.eye(dims) * kl_cov_jitter

	q = tfp.distributions.MultivariateNormalFullCovariance(
		loc=mean_q.transpose(),
		covariance_matrix=_cov_q,
		validate_args=False,
		allow_nan_stats=True,
	)
	p = tfp.distributions.MultivariateNormalFullCovariance(
		loc=mean_p.transpose(),
		covariance_matrix=_cov_p,
		validate_args=False,
		allow_nan_stats=True,
	)
	kl = tfd.kl_divergence(q, p, allow_nan_stats=False)
	return kl


def select_context_points(
	n_context_points: int,
	context_point_type: str,
	context_points_bound: Optional[List[int]],
	input_shape: List[int],
	x_batch: np.ndarray,
	rng_key: jnp.ndarray,
) -> jnp.ndarray:
	"""Select context points.

	:param n_context_points: integer, number of context points to select
	:param context_point_type: strategy to select context points
	:param context_points_bound: a list of two floats, usually [0.0, 1.0]
	:param input_shape: expected shape of context points, including batch dimension
	:param x_batch: input data of the current task
	:param rng_key: this random key will be fixed for different calls of select_context_points when
									n_marginals > 1.
	:return:
			context points.
	"""
	rng_key_1, rng_key_2 = jax.random.split(rng_key)
	permutation = jax.random.permutation(key=rng_key_1, x=x_batch.shape[0])
	x_batch_permuted = x_batch[permutation, :]
	# avoid modifying input variables
	input_shape = copy(input_shape)

	if context_point_type == "uniform_rand":
		input_shape[0] = n_context_points
		context_points = jax.random.uniform(
			rng_key_2,
			input_shape,
			jnp.float32,
			context_points_bound[0],
			context_points_bound[1],
		)
	elif "train_pixel_rand" in context_point_type:
		scale = jnp.float32(
			context_point_type.split("train_pixel_rand_", 1)[1].split("_", 1)[0]
		)
		n_context_points_sample = int((1 - scale) * n_context_points)
		n_context_points_train = n_context_points - n_context_points_sample
		training_samples = x_batch_permuted[:n_context_points_train]
		rng_key, _ = random.split(rng_key_2)
		if len(input_shape) == 4 and input_shape[-1] == 1:
			# Select random pixel values
			random_pixels = jax.random.choice(
				a=x_batch_permuted.flatten(),
				shape=(n_context_points_sample,),
				replace=False,
				key=rng_key,
			)
			pixel_samples = jnp.transpose(
				random_pixels * jnp.ones(input_shape), (3, 1, 2, 0)
			)
		elif len(input_shape) == 4 and input_shape[-1] > 1:
			image_dim = input_shape[1]
			num_channels = input_shape[-1]
			pixel_samples_list = []
			for channel in range(num_channels):
				# Select random pixel values for given channel
				random_pixels = jax.random.choice(
					a=x_batch_permuted[:, :, :, channel].flatten(),
					shape=(n_context_points_sample,),
					replace=False,
					key=rng_key,
				)
				_pixel_samples = jnp.transpose(
					random_pixels * jnp.ones([1, image_dim, image_dim, 1]),
					(3, 1, 2, 0)
				)
				pixel_samples_list.append(_pixel_samples)
			pixel_samples = jnp.concatenate(pixel_samples_list, axis=3)
		else:
			# Select random pixel values
			random_pixels = jax.random.choice(
				a=x_batch_permuted.flatten(),
				shape=(n_context_points_sample,),
				replace=False,
				key=rng_key,
			)[:, None]
			pixel_samples = random_pixels * jnp.ones(input_shape[-1])

		context_points = np.concatenate([pixel_samples, training_samples], 0)
	else:
		raise ValueError(
			f"context point select method specified ({context_point_type}) is not a valid setting."
		)
	return context_points


def _slice_cov_diag(cov: jnp.ndarray, index: int) -> jnp.ndarray:
	"""
	This function slices and takes diagonal

	index is for the output dimension
	"""
	ndims = len(cov.shape)
	if ndims == 2:
		cov_i = cov[:, index]
	elif ndims == 3:
		cov_i = cov[:, :, index]
	elif ndims == 4:
		cov_i = cov[:, index, :, index]
	else:
		raise ValueError("Posterior covariance shape not recognized.")
	return cov_i


@jit
def kl_diag_tfd(
	mean_q: jnp.ndarray, mean_p: jnp.ndarray, cov_q: jnp.ndarray,
	cov_p: jnp.ndarray
) -> jnp.ndarray:
	"""Return KL(q || p).

	:param mean_q: mean of Gaussian distribution q.
	:param mean_p: mean of Gaussian distribution p.
	:param cov_q: covariance of Gaussian distribution q, 2-D array.
	:param cov_q: covariance of Gaussian distribution p, 2-D array.
	:return:
		KL divergence.
	"""
	q = tfd.MultivariateNormalDiag(loc=mean_q, scale_diag=(cov_q ** 0.5))
	p = tfd.MultivariateNormalDiag(loc=mean_p, scale_diag=(cov_p ** 0.5))
	return tfd.kl_divergence(q, p)


def to_float_if_possible(x):
	try:
		return float(x)
	except ValueError:
		return x
