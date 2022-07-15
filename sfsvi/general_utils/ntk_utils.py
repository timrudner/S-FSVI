import pdb
from typing import Callable, Union

import haiku as hk
import jax
import numpy as np
import tree
from jax import numpy as jnp, jit
from tqdm import tqdm

from sfsvi.general_utils.custom_empirical_ntk import empirical_ntk_fn

NESTED_OR_ARRAY = Union[hk.Params, jnp.ndarray, np.ndarray]


def diag_ntk_for_loop(
    apply_fn: Callable, x: jnp.ndarray, params: hk.Params, sigma: hk.Params, diag=True
):
    """

    @param apply_fn: should have signature (params, inputs, **kwargs) and should return an np.ndarray outputs.
    @param x:
    @param params:
    @param sigma:
    @return:
        diag_ntk_sum_array: array of shape (batch_dim, output_dim)
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
    apply_fn: Callable, x: jnp.ndarray, params: hk.Params, sigma: hk.Params, diag=False
):
    """
    This function uses implicit implementation and neural tangent implementation.

    @param apply_fn: should have signature (params, inputs, **kwargs) and should return an np.ndarray outputs.
    @param x: input x
    @param params:
    @param sigma:
    @param diag:
    @param direct:
    @return:
        diag_ntk_sum_array: array of shape (batch_dim, output_dim) if diag==True else
         (batch_dim, output_dim, batch_dim, output_dim)
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


# @partial(jit, static_argnums=(0,3,4,))
def explicit_ntk(
    fwd_fn: Callable, params: hk.Params, sigma: hk.Params, diag, grad_flow_jacobian,
) -> jnp.ndarray:
    """
    Calculate J * diag(sigma) * J^T, where J is Jacobian of model with respect to model parameters
     using explicit implementation and einsum

    @param fwd_fn: a function that only takes in parameters and returns model output of shape (batch_dim, output_dim)
    @param params: the model parameters
    @param sigma: it has the same structure and array shapes as the parameters of model
    @param diag: if True, only calculating the diagonal of NTK
    @return:
        diag_ntk_sum_array: array of shape (batch_dim, output_dim) if diag==True else
         (batch_dim, output_dim, batch_dim, output_dim)
    """
    if grad_flow_jacobian:
        jacobian = jax.jacobian(fwd_fn)(params)
    else:
        jacobian = jax.jacobian(fwd_fn)(jax.lax.stop_gradient(params))

    # @jit
    def _get_diag_ntk(jac, sigma):
        # jac has shape (batch_dim, output_dim, params_dims...)
        # jac_2D has shape (batch_dim * output_dim, nb_params)
        batch_dim, output_dim = jac.shape[:2]
        jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
        # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D

        ## Diagonal final-layer:
        sigma_flatten = jnp.reshape(sigma, (-1,))
        # jac_sigma_product has the same shape as jac_2D
        jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)

        # diag_ntk has shape (batch_dim * output_dim,)
        if diag:
            ntk = jnp.einsum("ij,ji->i", jac_sigma_product, jac_2D.T)
            ntk = jnp.reshape(ntk, (batch_dim, output_dim))
            # ntk has shape (batch_dim, output_dim)
        else:
            ntk = jnp.matmul(jac_sigma_product, jac_2D.T)
            ntk = jnp.reshape(ntk, (batch_dim, output_dim, batch_dim, output_dim))
            # ntk has shape (batch_dim, output_dim, batch_dim, output_dim)
        return ntk

    diag_ntk = tree.map_structure(_get_diag_ntk, jacobian, sigma)
    diag_ntk_sum_array = jnp.stack(tree.flatten(diag_ntk), axis=0).sum(axis=0)
    return diag_ntk_sum_array
