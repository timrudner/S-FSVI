"""Stochastic networks used by S-FSVI."""
from functools import partial
from typing import Callable, Tuple, Union, List

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit

from sfsvi.models import custom_cnns
from sfsvi.models import custom_mlps

ACTIVATION_DICT = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}


class Model:
    def __init__(
        self,
        output_dim: int,
        architecture: Union[str, List[int]],
        no_final_layer_bias: bool,
        activation_fn: str = "relu",
        final_layer_variational: bool = False,
        dropout: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        :param output_dim: number of dimensions for the model output.
        :param architecture: architecture choice, e.g. different types of CNN,
            number of layers and hidden units for MLP.
            For instances,
                `[100, 100]` means a MLP of two layers of 100 hidden units each;
                `six_layers` means `SixLayers` CNN.
        :param no_final_layer_bias: if True, the last layer of neural networks
            doesn't have bias parameters.
        :param activation_fn: type of activation function.
        :param final_layer_variational: if True, then all the layers except
            the last layer don't have variance for variational parameters.
        :param dropout: if True, apply dropout.
        :param dropout_rate: dropout rate to apply if `dropout` is True.
        """
        self.output_dim = output_dim
        self.final_layer_variational = final_layer_variational
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.architecture = architecture
        self.no_final_layer_bias = no_final_layer_bias

        self.params_init = None
        self.params_eval_carry = None

        self.inner_layers_stochastic = not final_layer_variational

        if self.architecture == "vit":
            self.forward = self.make_forward_fn()()
        else:
            self.forward = hk.transform_with_state(self.make_forward_fn())

    @property
    def apply_fn(self) -> Callable:
        return self.forward.apply

    def make_forward_fn(self) -> Callable:
        """Create forward function that is returned by
        `hk.transform_with_state`."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 5))
    def predict_f(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        """Returns the logits."""
        return self.forward.apply(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
            stochastic=True,
            is_training=is_training,
        )[0]

    @partial(jit, static_argnums=(0, 5))
    def predict_y(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        """Returns the predicted class probabilities."""
        return jax.nn.softmax(
            self.predict_f(params, state, inputs, rng_key, is_training)
        )

    def predict_y_multisample(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        n_samples: int,
        is_training: bool,
    ):
        """Draw MC samples of predicted class probabilities."""
        return mc_sampling(
            fn=lambda _rng_key: self.predict_y(
                params, state, inputs, _rng_key, is_training
            ),
            n_samples=n_samples,
            rng_key=rng_key,
        )

    def predict_f_multisample_v1(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        n_samples: int,
        is_training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Draw MC samples of logits.

        :param params: model parameters.
        :param state: model state.
        :param inputs: a batch of input.
        :param rng_key: JAX random key.
        :param n_samples: number of MC samples.
        :param is_training: if True, apply dropout if necessary.

        :return:
            `n_samples` samples of logits, of shape
                (n_samples, inputs.shape[0], output_dimension).
            mean of logits, of shape (inputs.shape[0], output_dimension).
            variance of logits, of shape (inputs.shape[0], output_dimension).
        """
        pred_fn = lambda rng_key: self.predict_f(
            params, state, inputs, rng_key, is_training
        )

        return mc_sampling(
            fn=pred_fn,
            n_samples=n_samples,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0, 5, 6))
    def predict_f_multisample_v2_jitted(
        self,
        params,
        state,
        inputs,
        rng_key,
        n_samples: int,
        is_training: bool,
    ):
        """Jitted version of predict_f_multisample_v1"""
        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_f(
            params,
            state,
            inputs,
            rng_key,
            is_training,
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )
        preds_samples = predict_multisample_fn(rng_keys)

        preds_mean = preds_samples.mean(axis=0)
        preds_var = preds_samples.std(axis=0) ** 2
        return preds_samples, preds_mean, preds_var


class MLP(Model):
    def __init__(
        self,
        output_dim: int,
        architecture: str,
        no_final_layer_bias: bool,
        activation_fn: str = "relu",
        final_layer_variational: bool = False,
        dropout: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__(
            output_dim=output_dim,
            architecture=architecture,
            no_final_layer_bias=no_final_layer_bias,
            activation_fn=activation_fn,
            final_layer_variational=final_layer_variational,
            dropout=dropout,
            dropout_rate=dropout_rate,
        )

    def make_forward_fn(self):
        def forward_fn(inputs, rng_key, stochastic, is_training):
            _forward_fn = custom_mlps.CustomMLP(
                output_dim=self.output_dim,
                activation_fn=self.activation_fn,
                architecture=self.architecture,
                no_final_layer_bias=self.no_final_layer_bias,
                dropout=self.dropout,
                dropout_rate=self.dropout_rate,
                final_layer_variational=self.final_layer_variational,
            )
            return _forward_fn(inputs, rng_key, stochastic, is_training)

        return forward_fn


class CNN(Model):
    def __init__(
        self,
        output_dim: int,
        architecture: str,
        no_final_layer_bias: bool,
        activation_fn: str = "relu",
        final_layer_variational: bool = False,
        dropout: bool = False,
        dropout_rate: float = 0.0,
    ):

        super().__init__(
            output_dim=output_dim,
            architecture=architecture,
            no_final_layer_bias=no_final_layer_bias,
            activation_fn=activation_fn,
            final_layer_variational=final_layer_variational,
            dropout=dropout,
            dropout_rate=dropout_rate,
        )

    def make_forward_fn(self):
        if self.architecture == "six_layers":

            def forward_fn(inputs, rng_key, stochastic, is_training):
                _forward_fn = custom_cnns.SixLayers(
                    output_dim=self.output_dim,
                    activation_fn=self.activation_fn,
                    no_final_layer_bias=self.no_final_layer_bias,
                    final_layer_variational=self.final_layer_variational,
                )
                return _forward_fn(inputs, rng_key, stochastic, is_training)

        elif self.architecture == "omniglot_cnn":

            def forward_fn(inputs, rng_key, stochastic, is_training):
                _forward_fn = custom_cnns.OmniglotCNN(
                    output_dim=self.output_dim,
                    activation_fn=self.activation_fn,
                    final_layer_variational=self.final_layer_variational,
                )
                return _forward_fn(inputs, rng_key, stochastic, is_training)

        else:
            raise NotImplementedError(self.architecture)
        return forward_fn


def mc_sampling(
    fn: Callable, n_samples: int, rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs Monte Carlo sampling and returns the samples, the mean of samples
    and the variance of samples

    :param fn: a deterministic function that takes in a random key and returns
        one MC sample
    :param n_samples: number of MC samples
    :param rng_key: random key
    :return:
        `n_samples` samples of logits, of shape (n_samples, inputs.shape[0], output_dimension).
        mean of logits, of shape (inputs.shape[0], output_dimension).
        variance of logits, of shape (inputs.shape[0], output_dimension).
    """
    list_of_pred_samples = []
    for _ in range(n_samples):
        rng_key, subkey = jax.random.split(rng_key)
        output = fn(subkey)
        list_of_pred_samples.append(jnp.expand_dims(output, 0))
    preds_samples = jnp.concatenate(list_of_pred_samples, 0)
    preds_mean = preds_samples.mean(axis=0)
    preds_var = preds_samples.std(axis=0) ** 2
    return preds_samples, preds_mean, preds_var
