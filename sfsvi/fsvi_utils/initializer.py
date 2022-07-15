from typing import List

import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax

from sfsvi.fsvi_utils.utils_cl import select_context_points
from sfsvi.models.networks import CNN, Model
from sfsvi.models.networks import MLP as MLP
from sfsvi.models.haiku_mod import predicate_var, predicate_batchnorm
from sfsvi.fsvi_utils.objectives_cl import Objectives_hk
from sfsvi.fsvi_utils.prior import CLPrior


class Initializer:
    def __init__(
        self,
        hparams,
        input_shape: List[int],
        output_dim: int,
        stochastic_linearization: bool,
        n_train: int,
        n_batches: int,
        n_marginals: int,
        map_initialization: bool = False,
    ):
        """
        This class is just a place to put instantiation code, all the attributes should be immutable.

        @param output_dim: the task-specific number of output dimensions
        """
        self.hparams = hparams
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.stochastic_linearization = stochastic_linearization
        self.n_batches = n_batches
        self.n_train = n_train
        self.n_marginals = n_marginals

        self.map_initialization = map_initialization

        print(f"\n" f"MAP initialization: {self.map_initialization}")
        print(f"Full NTK computation: {self.hparams.full_ntk}")
        print(f"Stochastic linearization (posterior): {self.stochastic_linearization}")

    def reset_output_dim(self, output_dim: int, rng_key):
        self.output_dim = output_dim
        cl_prior = self.initialize_cl_prior()
        model = self.initialize_model(rng_key)
        return cl_prior, model

    def initialize_objective(self, model):
        return Objectives_hk(
            model=model,
            kl_scale=self.hparams.kl_scale,
            prior_type=self.hparams.prior_type,
            stochastic_linearization=self.stochastic_linearization,
            n_marginals=self.n_marginals,
            full_ntk=self.hparams.full_ntk,
        )

    def initialize_cl_prior(self):
        return CLPrior(
            prior_type=self.hparams.prior_type,
            output_dim=self.output_dim,
            full_ntk=self.hparams.full_ntk,
            prior_mean=self.hparams.prior_mean,
            prior_cov=self.hparams.prior_cov,
        )

    def initialize_model(
        self,
        rng_key,
    ):
        model = self._compose_model()
        init_fn, apply_fn = model.forward
        # INITIALIZE NETWORK STATE + PARAMETERS
        x_init = jnp.ones(self.input_shape)
        params_init, state = init_fn(
            rng_key, x_init, rng_key, model.stochastic_parameters, is_training=True
        )

        if self.map_initialization:
            # TODO: this logic doesn't handle state
            params_log_var = hk.data_structures.filter(predicate_var, params_init)
            params_batchnorm = hk.data_structures.filter(
                predicate_batchnorm, params_init
            )

            if "fashionmnist" in self.hparams.task:
                filename = (
                    "saved_models/fashionmnist/map/params_pickle_map_fashionmnist"
                )
            elif "cifar" in self.hparams.task:
                filename = "saved_models/cifar10/map/params_pickle_map_cifar10"
            else:
                raise ValueError("MAP parameter file not found.")

            # TODO: use absolute path instead of letting it depend on working directory?
            params_mean = np.load(filename, allow_pickle=True)

            params_init = hk.data_structures.merge(
                params_mean, params_log_var, params_batchnorm
            )

        return model, init_fn, apply_fn, state, params_init

    def _compose_model(self) -> Model:
        if "mlp" in self.hparams.model_type:
            network_class = MLP
        elif "cnn" or "resnet" in self.hparams.model_type:
            network_class = CNN
        else:
            raise ValueError("Invalid network type.")

        stochastic_parameters = (
            "mfvi" in self.hparams.model_type or "fsvi" in self.hparams.model_type
        )
        dropout = "dropout" in self.hparams.model_type
        if not dropout and self.hparams.dropout_rate > 0:
            raise ValueError("Dropout Rate not Zero in Non-Dropout Model.")

        # DEFINE NETWORK
        model = network_class(
            architecture=self.hparams.architecture,
            output_dim=self.output_dim,
            activation_fn=self.hparams.activation,
            regularization=self.hparams.regularization,
            stochastic_parameters=stochastic_parameters,
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

    def initialize_context_points_fn(self, x_ood=None):
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
