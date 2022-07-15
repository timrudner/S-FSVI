"""Utilties for computing approximate induced function distribution by
linearising stochastic neural networks around the mean of the parameter
Gaussian distribution."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import haiku as hk
import jax
import tree
from jax import eval_shape
from jax import jacobian
from jax import jit
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sfsvi.fsvi_utils.utils_cl import sigma_transform
from sfsvi.general_utils.ntk_utils import diag_ntk_for_loop
from sfsvi.general_utils.ntk_utils import neural_tangent_ntk
from sfsvi.models.haiku_mod import partition_params

tfd = tfp.distributions


def bnn_linearized_predictive_v2(
	apply_fn: Callable,
	params_mean: hk.Params,
	params_log_var: hk.Params,
	params_deterministic: hk.Params,
	state: hk.State,
	context_points: jnp.ndarray,
	rng_key: jnp.ndarray,
	stochastic_linearization: bool,
	full_ntk: bool,
	for_loop: bool = False,
	identity_cov: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""Return the mean and covariance of output of linearized BNN.

	:param apply_fn: apply function returned by `hk.transform_with_state`.
	:param params_mean: mean of Gaussian parameter distribution.
	:param params_log_var: log of variance of Gaussian parameter distribution.
	:param params_deterministic: parameters that are not stochastic.
	:param state: haiku state.
	:param context_points: a batch of context points for which to calculate
		approximate function distribution.
	:param rng_key: JAX random key.
	:param stochastic_linearization: if True, when computing the mean of
		Gaussian distribution of function output, use weights sampled from
		the variational distribution instead of the mean of the
		variational distribution.
	:param full_ntk: if True, take into account covariance between samples and
		between output dimensions in the calculation of NTK covariance matrix.
	:param for_loop: if True, compute the prior distribution by looping
			over all context points.
	:param identity_cov: if True, variances of all points are the same.
	:returns
		mean of function distribution, array of shape (batch_dim, output_dim)
		variance of function distributionarray of shape
			if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
			otherwise, (batch_dim, output_dim)
	"""
	is_training = True
	params = hk.data_structures.merge(params_mean, params_log_var,
									  params_deterministic)
	mean = apply_fn(
		params,
		state,
		None,
		context_points,
		rng_key,
		stochastic=stochastic_linearization,
		is_training=is_training,
	)[0]

	params_var = sigma_transform(params_log_var)

	if full_ntk:
		assert not identity_cov, "not implemented"
		predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
			apply_fn,
			context_points,
			params_log_var,
			params_deterministic,
			state,
			rng_key,
			stochastic_linearization,
			is_training=is_training
		)
		# the following line is equivalent to calculate J*diag(params_var)*J^T
		cov = get_ntk(
			predict_fn_for_empirical_ntk,
			delta_vjp_jvp,
			delta_vjp,
			params_mean,
			params_var,
		)
	else:
		if identity_cov:
			cov = jnp.ones_like(mean)
		else:
			def predict_f(params_mean, x):
				params = hk.data_structures.merge(params_mean, params_log_var,
												  params_deterministic)
				return apply_fn(
					params,
					state,
					None,
					x,
					rng_key,
					stochastic=stochastic_linearization,
					is_training=is_training,
				)[0]

			renamed_params_var = map_variable_name(
				params_var, lambda n: f"{n.split('_')[0]}_mu"
			)

			if for_loop:
				# NOTE: if using this option, then do not jit the outside function, otherwise the compilation is slow
				# large sample option, only use this if small_samples solution gives OOM error
				cov = diag_ntk_for_loop(
					apply_fn=predict_f,
					x=context_points,
					params=params_mean,
					sigma=renamed_params_var,
				)
			else:
				cov = neural_tangent_ntk(
					apply_fn=predict_f,
					x=context_points,
					params=params_mean,
					sigma=renamed_params_var,
					diag=True
				)
	return mean, cov


def induced_prior_fn_refactored(
	apply_fn: Callable,
	params: hk.Params,
	state: hk.State,
	context_points: Dict[int, jnp.ndarray],
	rng_key: jnp.ndarray,
	task_id: int,
	stochastic_linearization: bool,
	full_ntk: bool = False,
	identity_cov: bool = False,
	for_loop: bool = False
) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
	"""Return the mean and covariance of output of linearized BNN.

	:param apply_fn: apply function returned by `hk.transform_with_state`.
	:param params: parameters of model.
	:param state: state of model.
	:param context_points: a batch of context points for which to calculate
		approximate function distribution.
	:param rng_key: JAX random key.
	:param task_id: task id of the current task.
	:param stochastic_linearization: if True, when computing the mean of
		Gaussian distribution of function output, use weights sampled from
		the variational distribution instead of the mean of the
		variational distribution.
	:param full_ntk: if True, take into account covariance between samples and
		between output dimensions in the calculation of NTK covariance matrix.
	:param identity_cov: if True, variances of all points are the same.
	:param for_loop: if True, compute the prior distribution by looping
			over all context points.
	:returns
		mean of function distribution, array of shape (batch_dim, output_dim)
		variance of function distributionarray of shape
			if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
			otherwise, (batch_dim, output_dim)

	Return mean and covariance matrix on context points for all tasks whose
	task id is equal or smaller than `task_id`
	"""
	params_mean_prior, params_log_var_prior, params_deterministic_prior = partition_params(
		params)
	prior_means, prior_covs = {}, {}
	task_ids = sorted(context_points.keys())
	assert max(task_ids) <= task_id
	for t_id in task_ids:
		x_context = context_points[t_id]
		prior_mean, prior_cov = bnn_linearized_predictive_v2(
			apply_fn,
			params_mean_prior,
			params_log_var_prior,
			params_deterministic_prior,
			state,
			x_context,
			rng_key,
			stochastic_linearization,
			full_ntk,
			identity_cov=identity_cov,
			for_loop=for_loop,
		)
		prior_means[t_id] = prior_mean
		prior_covs[t_id] = prior_cov
	return prior_means, prior_covs


def convert_predict_f_only_mean(
	apply_fn,
	inputs: jnp.ndarray,
	params_log_var: hk.Params,
	params_deterministic: hk.Params,
	state: hk.State,
	rng_key: jnp.ndarray,
	stochastic_linearization: bool,
	is_training: bool,
) -> Callable[[hk.Params], jnp.ndarray]:
	"""Return a function that takes the mean of parameter distribution and
	returns the logits.

	:param apply_fn: apply function returned by `hk.transform_with_state`.
	:param inputs: input array.
	:param params_log_var: log of variance of Gaussian parameter distribution.
	:param params_deterministic: parameters that are not stochastic.
	:param state: haiku state.
	:param rng_key: JAX random key.
	:param stochastic_linearization: if True, when computing the mean of
		Gaussian distribution of function output, use weights sampled from
		the variational distribution instead of the mean of the
		variational distribution.
	:param is_training: whether it is in training mode.
	:return:
	"""

	def predict_f_only_mean(params_mean):
		params = hk.data_structures.merge(
			params_mean,
			params_log_var,
			params_deterministic
		)
		return apply_fn(
			params,
			state,
			None,
			inputs,
			rng_key,
			stochastic=stochastic_linearization,
			is_training=is_training,
		)[0]

	return predict_f_only_mean


@partial(jit, static_argnums=(0,))
def delta_vjp(
	predict_fn,
	params_mean: hk.Params,
	params_var: hk.Params,
	delta,
):
	vjp_tp = jax.vjp(predict_fn, params_mean)[1](delta)
	renamed_params_var = map_variable_name(
		params_var, lambda n: f"{n.split('_')[0]}_mu"
	)
	return (
		tree.map_structure(lambda x1, x2: x1 * x2, renamed_params_var,
						   vjp_tp[0]),)


@partial(jit, static_argnums=(0, 1,))
def delta_vjp_jvp(
	predict_fn,
	delta_vjp: Callable,
	params_mean: hk.Params,
	params_var: hk.Params,
	delta: jnp.ndarray
):
	delta_vjp_ = partial(delta_vjp, predict_fn, params_mean, params_var)
	return jax.jvp(predict_fn, (params_mean,), delta_vjp_(delta))[1]


@partial(jit, static_argnums=(0, 1, 2))
def get_ntk(
	predict_fn,
	delta_vjp_jvp: Callable,
	delta_vjp: Callable,
	params_mean: hk.Params,
	params_var: hk.Params,
) -> jnp.ndarray:
	"""Returns the covariance matrix of function distribution."""
	predict_struct = eval_shape(predict_fn, params_mean)
	fx_dummy = jnp.ones(predict_struct.shape, predict_struct.dtype)
	delta_vjp_jvp_ = partial(
		delta_vjp_jvp, predict_fn, delta_vjp, params_mean, params_var
	)
	gram_matrix = jacobian(delta_vjp_jvp_)(fx_dummy)
	return gram_matrix


def map_variable_name(params: hk.Params, fn: Callable) -> hk.Params:
	"""Change parameters names."""
	params = hk.data_structures.to_mutable_dict(params)
	for module in params:
		params[module] = {
			fn(var_name): array for var_name, array in params[module].items()
		}
	return hk.data_structures.to_immutable_dict(params)
