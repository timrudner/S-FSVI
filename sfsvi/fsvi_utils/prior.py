from functools import partial
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from sfsvi.fsvi_utils import utils_linearization
from sfsvi.fsvi_utils.utils_cl import generate_4d_identity_cov


class CLPrior:
    def __init__(
        self,
        prior_type,
        output_dim: int,
        full_ntk: bool,
        prior_mean=float,
        prior_cov=float,
    ):
        """
        @param output_dim: the task-specific number of output dimensions
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
        jit_prior = True,
        identity_cov = False,
        for_loop = False,
        prior_type: str = None,
    ) -> Callable[
        [Dict[int, jnp.ndarray]], Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]
    ]:
        """

        @predict_f_deterministic: function to do forward pass
        @param prior_mean: example: "0.0"
        @param prior_cov: example: "0.0"
        @return:
            prior_fn: a function that takes in a dict that maps from task id to context point points,
                and returns two dicts that map from task id to mean and covariance respectively.
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
                # it is always the latest task that needs random context points
                # TODO make this logic easier to understand
                max_key = max(list(context_points.keys()))
                x_len = len(context_points[max_key])
                shape_mean = (x_len, self.output_dim)
                prior_means = {max_key: jnp.ones(shape_mean) * prior_mean}
                # the reason to use `generate_4d_identity_cov` is to make sure the covariance is well conditioned
                # when setting full_cov = True (i.e. covariance between data points per task is taken into account)
                prior_covs = {max_key: generate_4d_identity_cov(*shape_mean) * prior_cov}
                return prior_means, prior_covs

        else:
            raise NotImplementedError(prior_type)
        return prior_fn
