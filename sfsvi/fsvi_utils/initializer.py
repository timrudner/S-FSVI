"""Helper class for initialising components for the main training loop of
S-FSVI."""
from typing import List, Callable

import jax.numpy as jnp
import optax

from sfsvi.fsvi_utils.objectives_cl import Objectives_hk
from sfsvi.fsvi_utils.prior import CLPrior
from sfsvi.fsvi_utils.utils_cl import select_context_points
from sfsvi.general_utils.log import Hyperparameters
from sfsvi.models.networks import CNN
from sfsvi.models.networks import MLP as MLP
from sfsvi.models.networks import Model


class Initializer:
    """Helper class for initialising components for a training loop (e.g.
    model, optimiser)."""

    def __init__(
        self,
        hparams: Hyperparameters,
        input_shape: List[int],
        output_dim: int,
        stochastic_linearization: bool,
        n_marginals: int,
    ):
        """
        :param hparams: hyperparameters.
        :param input_shape: shape of input data.
        :param output_dim: the number of output dimensions.
        :param stochastic_linearization: if True, when computing the mean of
            Gaussian distribution of function output, use weights sampled from
            the variational distribution instead of the mean of the
            variational distribution.
        """
        self.hparams = hparams
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.stochastic_linearization = stochastic_linearization
        self.n_marginals = n_marginals
        print(f"Full NTK computation: {self.hparams.full_ntk}")
        print(f"Stochastic linearization (posterior): {self.stochastic_linearization}")

    def reset_output_dim(self, output_dim: int, rng_key: jnp.ndarray) -> ...:
        """Recreate prior function and model given the new output dimension.

        This is useful when we want to gradually adding the output heads.

        :param output_dim: the new output dimensions.
        :param rng_key: JAX random key.
        :return:
            CL prior;
            model.
        """
        self.output_dim = output_dim
        cl_prior = self.initialize_cl_prior()
        model_tuple = self.initialize_model(rng_key)
        return cl_prior, model_tuple

    def initialize_objective(self, model: Model) -> Objectives_hk:
        return Objectives_hk(
            model=model,
            kl_scale=self.hparams.kl_scale,
            stochastic_linearization=self.stochastic_linearization,
            n_marginals=self.n_marginals,
            full_ntk=self.hparams.full_ntk,
        )

    def initialize_cl_prior(self) -> CLPrior:
        return CLPrior(
            prior_type=self.hparams.prior_type,
            output_dim=self.output_dim,
            full_ntk=self.hparams.full_ntk,
            prior_mean=self.hparams.prior_mean,
            prior_cov=self.hparams.prior_cov,
        )

    def initialize_model(self, rng_key: jnp.ndarray) -> ...:
        model = self._compose_model()
        init_fn, apply_fn = model.forward
        # INITIALIZE NETWORK STATE + PARAMETERS
        x_init = jnp.ones(self.input_shape)
        params_init, state = init_fn(rng_key, x_init, rng_key, True, is_training=True)

        return model, init_fn, apply_fn, state, params_init

    def _compose_model(self) -> Model:
        if "mlp" in self.hparams.model_type:
            network_class = MLP
        elif "cnn" or "resnet" in self.hparams.model_type:
            network_class = CNN
        else:
            raise ValueError("Invalid network type.")

        dropout = "dropout" in self.hparams.model_type
        if not dropout and self.hparams.dropout_rate > 0:
            raise ValueError("Dropout Rate not Zero in Non-Dropout Model.")

        # DEFINE NETWORK
        model = network_class(
            architecture=self.hparams.architecture,
            output_dim=self.output_dim,
            activation_fn=self.hparams.activation,
            final_layer_variational=self.hparams.final_layer_variational,
            dropout=dropout,
            dropout_rate=self.hparams.dropout_rate,
            no_final_layer_bias=self.hparams.no_final_layer_bias,
        )
        return model

    def initialize_optimizer(
        self,
        learning_rate=None,
    ) -> optax.GradientTransformation:
        _learning_rate = (
            self.hparams.learning_rate if learning_rate is None else learning_rate
        )
        opt = optax.adam(_learning_rate)
        return opt

    def initialize_context_points_fn(self) -> Callable:
        """Initialise context points."""
        def context_point_fn(x_batch, rng_key, n_context_points=None):
            if n_context_points is None:
                n_context_points = self.hparams.n_context_points
            return select_context_points(
                n_context_points=n_context_points,
                context_point_type=self.hparams.context_point_type,
                context_points_bound=self.hparams.context_points_bound,
                input_shape=self.input_shape,
                x_batch=x_batch,
                rng_key=rng_key,
            )

        return context_point_fn
