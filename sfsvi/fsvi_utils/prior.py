"""Prior function distribution of S-FSVI."""
from functools import partial
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from sfsvi.fsvi_utils import utils_linearization
from sfsvi.fsvi_utils.utils_cl import generate_4d_identity_cov


class CLPrior:
    """Prior function distribution of S-FSVI."""

    def __init__(
        self,
        prior_type: str,
        output_dim: int,
        full_ntk: bool,
        prior_mean: str,
        prior_cov: str,
    ):
        """
        :param prior_type: type of the prior.
            "bnn_induced": the prior is induced function distribution on the
                stochastic neural networks.
            "fixed": the prior is a GP with fixed mean, diagonal
                covariance matrix for any set of input points.
        :param output_dim: the number of output dimensions.
        :param full_ntk: if True, take into account covariance between samples
            and between output dimensions in the calculation of NTK covariance
            matrix.
        :param prior_mean: if the `prior_type` is `fixed`, `float(prior_mean)`
            is the mean of Gaussian distribution at any input point.
        :param prior_cov: if the `prior_type` is `fixed`, `float(prior_cov)`
            is the variance of Gaussian distribution at any input point.
        """
        self.prior_type = prior_type
        self.output_dim = output_dim
        self.full_ntk = full_ntk
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.stochastic_linearization_prior = False
        print(f"Full NTK computation: {self.full_ntk}")
        print(
            f"Stochastic linearization (prior): {self.stochastic_linearization_prior}"
            f"\n"
        )

    def make_prior_fn(
        self,
        apply_fn: Callable,
        state: hk.State,
        params: hk.Params,
        rng_key,
        task_id: int,
        jit_prior: bool = True,
        identity_cov: bool = True,
        for_loop: bool = True,
        prior_type: str = None,
    ) -> Callable[
        [Dict[int, jnp.ndarray]], Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]
    ]:
        """Return a function for computing the prior function distribution
        on any given points.

        :param apply_fn: apply_fn returned by `hk.transform_with_state`.
        :param state: the haiku state.
        :param params: the haiku parameters.
        :param rng_key: JAX random key.
        :param task_id: the task id of the current task.
        :param jit_prior: if True, jit the prior function.
        :param identity_cov: if True, variances of all points are the same.
        :param for_loop: if True, compute the prior distribution by looping
            over all context points.
        :param prior_type: type of the prior.
            "bnn_induced": the prior is induced function distribution on the
                stochastic neural networks.
            "fixed": the prior is a GP with fixed mean, diagonal
                covariance matrix for any set of input points.
        :return:
            a function for computing the prior function distribution
                on any given points.
        """
        if prior_type is None:
            prior_type = "fixed" if task_id == 0 else self.prior_type

        prior_mean, prior_cov = (
            jnp.float32(self.prior_mean),
            jnp.float32(self.prior_cov),
        )

        if prior_type == "bnn_induced":
            rng_key0, _ = jax.random.split(rng_key)

            # prior_fn is a function of context_points
            prior_fn = partial(
                utils_linearization.induced_prior_fn_refactored,
                apply_fn=apply_fn,
                params=params,
                state=state,
                rng_key=rng_key0,
                task_id=task_id,
                stochastic_linearization=self.stochastic_linearization_prior,
                full_ntk=self.full_ntk,
                for_loop=for_loop,
            )
            if jit_prior and not identity_cov:
                prior_fn = jax.jit(prior_fn)

        elif prior_type == "fixed":

            def prior_fn(context_points: Dict):
                # It is always the latest task that needs random context points.
                max_key = max(list(context_points.keys()))
                x_len = len(context_points[max_key])
                shape_mean = (x_len, self.output_dim)
                prior_means = {max_key: jnp.ones(shape_mean) * prior_mean}
                # The reason to use `generate_4d_identity_cov` is to make sure
                # the covariance is well conditioned when setting
                # full_cov = True (i.e. covariance between data points per task
                # is taken into account)
                prior_covs = {
                    max_key: generate_4d_identity_cov(*shape_mean) * prior_cov
                }
                return prior_means, prior_covs

        else:
            raise NotImplementedError(prior_type)
        return prior_fn
