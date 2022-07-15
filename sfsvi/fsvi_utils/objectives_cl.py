from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk

from sfsvi.models.networks import Model
from sfsvi.fsvi_utils import utils_linearization
from sfsvi.fsvi_utils.utils_cl import kl_diag_tfd
from sfsvi.fsvi_utils.utils_cl import _slice_cov_diag
from sfsvi.fsvi_utils.utils_cl import kl_full_cov
from sfsvi.fsvi_utils.utils_cl import TUPLE_OF_TWO_TUPLES
from sfsvi.models.haiku_mod import partition_params


@partial(jit, static_argnums=(0,3,))
def compute_scale(kl_scale: str, inputs: jnp.ndarray, n_context_points: int, n_marginals: int) -> float:
    if kl_scale == "none":
        scale = 1.0
    elif kl_scale == "equal":
        scale = inputs.shape[0] / n_context_points
    elif kl_scale == "normalized":
        if n_marginals > 1:
            scale = 1.0 / (n_context_points // n_marginals)
        else:
            scale = 1.0 / n_context_points
    else:
        scale = jnp.float32(kl_scale)
    return scale


class Objectives_hk:
    def __init__(
        self,
        model: Model,
        kl_scale: str,
        prior_type,
        stochastic_linearization,
        n_marginals,
        full_ntk: bool,
    ):
        self.model = model
        self.kl_scale = kl_scale
        self.prior_type = prior_type
        self.stochastic_linearization = stochastic_linearization
        self.full_ntk = full_ntk
        self.n_marginals = n_marginals


    def _crossentropy_log_likelihood_multihead(
        self,
        preds_f_samples: jnp.ndarray,
        targets: jnp.ndarray,
        min_dim: int,
        max_dim: int,
    ) -> jnp.ndarray:
        """
        Returns the Monte Carlo estimated log likelihood (or negative cross entropy)

        @param preds_f_samples: array of shape (nb_samples, batch_dim, output_dim)
        @param targets: one-hot encoded labels, array of shape (batch_dim, output_dim)
        @param min_dim:
        @param max_dim:
        @return:
            a scalar array
        """
        # log_preds_y_samples has shape (nb_samples, batch_dim, output_dim)
        log_preds_y_samples = jax.ops.index_update(
            jnp.zeros_like(preds_f_samples),  # the base array
            jax.ops.index[:, :, min_dim:max_dim],  # which indices we're accessing
            jax.nn.log_softmax(preds_f_samples[:, :, min_dim:max_dim], axis=-1),
        )  # what we're putting at those indices
        # targets is broadcasted in the sample dimension
        log_likelihood = jnp.mean(
            jnp.sum(jnp.sum(targets * log_preds_y_samples, axis=-1), axis=-1), axis=0
        )
        return log_likelihood

    def _elbo_fsviassification_multihead(
        self,
        params: hk.Params,
        state: hk.State,
        prior_means: Dict[int, jnp.ndarray],
        prior_covs: Dict[int, jnp.ndarray],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        context_points: Dict[int, jnp.ndarray],
        rng_key: jnp.ndarray,
        range_dims_per_task: TUPLE_OF_TWO_TUPLES,
        task_id: int,
        n_samples: int,
        is_training: bool,
        loss_type: int,
    ):
        """

        @param params: parameters of the BNN
        @param state: state of the BNN
        @param prior_means: a list of prior mean array of shape (batch_dim, output_dim)
        @param prior_covs: a list of prior mean array of shape (batch_dim, output_dim, batch_dim, output_dim)
        @param inputs: input data of the current task, used to calculate expected log likelihood
        @param targets: labels, array of shape (batch_dim, output_dim), used to calculate expected log likelihood
        @param context_points: an array with context points from all task, or a dictionary that maps
            task id to the context points
        @param rng_key: random seed controlling the Monte Carlo estimation of expected log likelihood
        @param range_dims_per_task:
        @return:
            elbo: scalar
            log_likelihood: scalar
            kl: scalar
            scale: scalar
        """
        _check_input(
            prior_means=prior_means,
            prior_covs=prior_covs,
            context_points=context_points,
            task_id=task_id,
        )
        params_mean, params_log_var, params_deterministic = partition_params(params)
        # TODO: parallelize this loop
        kl = 0
        for t_id in sorted(prior_means.keys()):
            prior_mean, prior_cov = prior_means[t_id], prior_covs[t_id]
            context_points_task = context_points[t_id]
            min_dim, max_dim = range_dims_per_task[t_id]

            mean, cov = utils_linearization.bnn_linearized_predictive_v2(
                self.model.apply_fn,
                params_mean,
                params_log_var,
                params_deterministic,
                state,
                context_points_task,
                rng_key,
                self.stochastic_linearization,
                self.full_ntk,
            )

            kl += kl_divergence_min_max_dim(
                mean, prior_mean, cov, prior_cov, min_dim, max_dim, 1e-6, self.full_ntk
            )

        preds_f_samples, _, _ = self.model.predict_f_multisample_jitted(
            params, params, state, inputs, rng_key, n_samples, is_training
        )
        min_dim, max_dim = range_dims_per_task[task_id] if task_id != -1 \
            else (range_dims_per_task[0][0], range_dims_per_task[-1][1])
        log_likelihood = self.crossentropy_log_likelihood_multihead(
            preds_f_samples, targets, min_dim, max_dim
        )

        n_context_points = sum([array.shape[0] for array in context_points.values()])
        scale = compute_scale(self.kl_scale, inputs, n_context_points, self.n_marginals)
        if loss_type == 1:
            elbo = log_likelihood - scale * kl
        elif loss_type == 2:
            elbo = (log_likelihood - scale * kl) / inputs.shape[0]
        else:
            raise NotImplementedError(loss_type)

        return elbo, log_likelihood, kl, scale

    @partial(jit, static_argnums=(0, 3, 4))
    def crossentropy_log_likelihood_multihead(
        self, preds_f_samples, targets, min_dim, max_dim
    ) -> jnp.ndarray:
        return self._crossentropy_log_likelihood_multihead(
            preds_f_samples, targets, min_dim, max_dim
        )

    @partial(jit, static_argnums=(0, 9, 10, 11, 12, 13))
    def nelbo_fsviassification_multihead(
        self,
        params,
        state,
        prior_means,
        prior_covs,
        inputs,
        targets,
        context_points,
        rng,
        range_dims_per_task,
        task_id,
        n_samples: int,
        is_training: bool,
        loss_type: int,
    ):
        elbo, log_likelihood, kl, scale = self._elbo_fsviassification_multihead(
            params,
            state,
            prior_means,
            prior_covs,
            inputs,
            targets,
            context_points,
            rng,
            range_dims_per_task,
            task_id,
            n_samples=n_samples,
            is_training=is_training,
            loss_type=loss_type,
        )
        return (
            -elbo,
            {"log_likelihood": log_likelihood, "kl": kl, "scale": scale, "elbo": elbo},
        )


@partial(jit, static_argnums=(4, 5, 6, 7))
def kl_divergence_min_max_dim(mean_q, mean_p, cov_q, cov_p, min_dim: int, max_dim: int, noise: float = 1e-6,
                              full_cov: bool = False):
    """
    Return the sum of KL(q || p) calculated for each output dimension between min_dim and max_dim
    @param mean_q: array of shape (batch_dim, output_dim)
    @param mean_p: array of shape (batch_dim, output_dim)
    """
    kl = 0
    # TODO: parallelize this loop
    for i in range(min_dim, max_dim):
        mean_q_i = jnp.squeeze(mean_q[:, i])
        mean_p_i = jnp.squeeze(mean_p[:, i])
        cov_q_i = _slice_cov_diag(cov=cov_q, index=i)
        cov_p_i = _slice_cov_diag(cov=cov_p, index=i)
        if not full_cov:
            if len(cov_q_i.shape) != 1:
                cov_q_i = jnp.diag(cov_q_i)
            if len(cov_p_i.shape) != 1:
                cov_p_i = jnp.diag(cov_p_i)
            kl += kl_diag_tfd(mean_q_i, mean_p_i, cov_q_i, cov_p_i)
        else:
            noise_matrix = jnp.eye(cov_q_i.shape[0]) * noise
            cov_q_i += noise_matrix
            cov_p_i += noise_matrix
            kl += kl_full_cov(mean_q_i, mean_p_i, cov_q_i, cov_p_i)
    return kl


def _check_input(
    prior_means: Dict[int, jnp.ndarray],
    prior_covs: Dict[int, jnp.ndarray],
    context_points: Dict[int, jnp.ndarray],
    task_id: int,
):
    def _kvs(d: Dict):
        keys = sorted(d.keys())
        num_context_samples = [d[k].shape[0] for k in keys]
        assert keys[-1] <= task_id
        return keys, num_context_samples

    assert _kvs(prior_means) == _kvs(prior_covs) == _kvs(context_points)
