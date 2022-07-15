"""Stochastic multi-layer perceptron."""
from typing import Callable, List

import haiku as hk
import jax.numpy as jnp

from sfsvi.models.haiku_mod import KeyHelper
from sfsvi.models.haiku_mod import DenseStochastic


class CustomMLP(hk.Module):
    def __init__(
        self,
        output_dim: int,
        activation_fn: Callable,
        no_final_layer_bias: bool,
        dropout: bool,
        dropout_rate: float,
        final_layer_variational: bool,
        architecture: List[int],
        init_logvar_minval: float = -10,
        init_logvar_maxval: float = -8,
    ):
        """
        :param output_dim: the number of output dimensions.
        :param activation_fn: activation function.
        :param no_final_layer_bias: if True, do not include bias parameters
            for the final layer.
        :param dropout: if True, apply dropout.
        :param dropout_rate: dropout rate if applying dropout.
        :param final_layer_variational: if True, then all the layers except
            the last layer don't have variance for variational parameters.
        :param architecture: number of layers and hidden units for MLP.
            For example, `[100, 100]` means an MLP of two layers of 100 hidden
            units each.
        :param init_logvar_minval: lower bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param init_logvar_maxval: upper bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        """
        super().__init__()
        self.activation_fn = activation_fn
        self.output_dim = output_dim
        self.architecture = architecture
        self.no_final_layer_bias = no_final_layer_bias
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        inner_layers_stochastic = not final_layer_variational

        self.layers = []
        for unit in architecture[:-1]:
            self.layers.append(
                DenseStochastic(
                    unit,
                    init_logvar_minval=init_logvar_minval,
                    init_logvar_maxval=init_logvar_maxval,
                    stochastic_parameters=inner_layers_stochastic,
                )
            )

        # PENULTIMATE LAYER
        self.layers.append(
            DenseStochastic(
                architecture[-1],
                init_logvar_minval=init_logvar_minval,
                init_logvar_maxval=init_logvar_maxval,
                stochastic_parameters=inner_layers_stochastic,
            )
        )

        # FINAL LAYER
        self.layers.append(
            DenseStochastic(
                self.output_dim,
                init_logvar_minval=init_logvar_minval,
                init_logvar_maxval=init_logvar_maxval,
                stochastic_parameters=True,
                with_bias=not self.no_final_layer_bias,
                name="linear_final",
            )
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        stochastic: bool,
        is_training: bool,
    ) -> jnp.ndarray:
        """Forward pass of the model.

        :param inputs: input data.
        :param rng_key: JAX random key.
        :param stochastic: if True, instantiate stochastic layers instead of
            deterministic layers.
        :param is_training: if True, apply dropout if it is set.
        :return:
            logits.
        """
        kh = KeyHelper(rng_key)
        out = inputs

        for l in range(len(self.layers) - 2):
            out = self.layers[l](out, kh.next_key(), stochastic)
            out = self.activation_fn(out)

        # PENULTIMATE LAYER
        out = self.activation_fn(self.layers[-2](out, kh.next_key(), stochastic))

        # FINAL LAYER
        out = self.layers[-1](out, rng_key, stochastic)

        return out
