"""Utilities for calculating the covariance of function output distribution."""
from typing import Callable
from typing import Union

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm

from sfsvi.general_utils.custom_empirical_ntk import empirical_ntk_fn

NESTED_OR_ARRAY = Union[hk.Params, jnp.ndarray, np.ndarray]


def diag_ntk_for_loop(
    apply_fn: Callable, x: jnp.ndarray, params: hk.Params, sigma: hk.Params, diag: bool = True,
):
    """Calculate the diagonal of `J * sigma * J^T`, where
        `J` is Jacobian of the function that takes model parameters and returns
            logits with respect to a batch of data `x`;
        `sigma` is the variance of parameter.

    :param apply_fn: apply_fn returned by `hk.transform_with_state`.
    :param x: a batch of input data points.
    :param params: model parameters.
    :param sigma: variance of parameter.
    :param diag: if True, calculate the diagonal of `J * sigma * J^T` instead
        of the full matrix.
    :return:
        diagonal of `J * sigma * J^T`, of shape (batch_dim, output_dim).
    """
    assert diag
    kernel_fn = empirical_ntk_fn(
        f=apply_fn, trace_axes=(), diagonal_axes=(-1,), vmap_axes=0, implementation=2,
    )

    @jax.jit
    def _one_iteration(one_sample_x: jnp.ndarray):
        return kernel_fn(one_sample_x, None, params, sigma)

    covs = []
    for i in tqdm(range(x.shape[0]), desc="using neural-tangent repo implementation for per-sample ntk evaluation",
                  disable=True):
        covs.append(_one_iteration(x[i : i + 1]))
    cov = jnp.squeeze(jnp.array(covs), axis=(1, 2))
    return cov


def neural_tangent_ntk(
    apply_fn: Callable, x: jnp.ndarray, params: hk.Params, sigma: hk.Params, diag: bool = False
):
    """This function uses implicit implementation of the neural tangent repo.

    https://github.com/google/neural-tangents

    :param apply_fn: apply_fn returned by `hk.transform_with_state`.
    :param x: a batch of input data points.
    :param params: model parameters.
    :param sigma: variance of parameter.
    :param diag: if True, calculate the diagonal of `J * sigma * J^T` instead
        of the full matrix.
    :return:
        diagonal of `J * sigma * J^T`,
            of shape (batch_dim, output_dim) if `diag` is True, otherwise
            of shape (batch_dim, output_dim, batch_dim, output_dim).
    """
    kernel_fn = empirical_ntk_fn(
        f=apply_fn,
        trace_axes=(),
        diagonal_axes=(-1,) if diag else (),
        vmap_axes=0,
        implementation=2,
    )
    cov = kernel_fn(x, None, params, sigma)
    # reshaped_cov has shape (batch_dim, output_dim, batch_dim, output_dim)
    if diag:
        # cov has shape (batch_dim, batch_dim, output_dim)
        # reshaped_cov has shape (batch_dim, output_dim)
        reshaped_cov = jax.vmap(jnp.diag, in_axes=2, out_axes=1)(cov)
    else:
        # cov has shape (batch_dim, batch_dim, output_dim, output_dim)
        reshaped_cov = jnp.transpose(cov, (0, 2, 1, 3))
    return reshaped_cov
