"""Stochastic convolutional neural networks."""
from typing import Callable
from typing import Sequence

import haiku as hk
import jax.numpy as jnp

from sfsvi.models.haiku_mod import KeyHelper
from sfsvi.models.haiku_mod import Conv2DStochastic
from sfsvi.models.haiku_mod import DenseStochastic


class SixLayers(hk.Module):
    """CNN with 4 conv layers and 2 fc layers.

    https://www.arxiv-vanity.com/papers/1703.04200/
    """

    def __init__(
        self,
        output_dim: int,
        activation_fn: Callable,
        no_final_layer_bias: bool,
        final_layer_variational: bool,
        channels: Sequence[int] = (32, 32, 64, 64),
        fc_dim: int = 512,
        init_logvar_minval: float = -10,
        init_logvar_maxval: float = -8,
    ):
        """
        :param output_dim: the number of output dimensions.
        :param activation_fn: activation function.
        :param no_final_layer_bias: if True, do not include bias parameters
            for the final layer.
        :param final_layer_variational: if True, then all the layers except
            the last layer don't have variance for variational parameters.
        :param channels: number of channels for each convolution layer.
        :param fc_dim: number of hidden units for fully connected layer.
        :param init_logvar_minval: lower bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param init_logvar_maxval: upper bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        """
        super().__init__()
        self.activation_fn = activation_fn
        self.no_final_layer_bias = no_final_layer_bias

        self.max_pool = hk.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
        )

        inner_layers_stochastic = not final_layer_variational

        self.conv = []
        for channels_i in channels:
            conv_i = Conv2DStochastic(
                output_channels=channels_i,
                kernel_shape=3,
                stochastic_parameters=inner_layers_stochastic,
                init_logvar_minval=init_logvar_minval,
                init_logvar_maxval=init_logvar_maxval,
            )
            self.conv.append(conv_i)

        self.fc_1 = DenseStochastic(
            output_size=fc_dim,
            stochastic_parameters=inner_layers_stochastic,
            init_logvar_minval=init_logvar_minval,
            init_logvar_maxval=init_logvar_maxval,
        )
        self.fc_2 = DenseStochastic(
            output_size=output_dim,
            stochastic_parameters=True,
            name="linear_final",
            init_logvar_minval=init_logvar_minval,
            init_logvar_maxval=init_logvar_maxval,
            with_bias=not no_final_layer_bias,
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

        for i in range(2):
            out = self.conv[i](out, kh.next_key(), stochastic)
            out = self.activation_fn(out)
        out = self.max_pool(out)

        if is_training:
            out = hk.dropout(kh.next_key(), 0.25, out)

        for i in range(2, 4):
            out = self.conv[i](out, kh.next_key(), stochastic)
            out = self.activation_fn(out)
        out = self.max_pool(out)

        out = jnp.moveaxis(out, source=3, destination=1)
        out = out.reshape(inputs.shape[0], -1)

        if is_training:
            out = hk.dropout(kh.next_key(), 0.25, out)
        out = self.fc_1(out, kh.next_key(), stochastic)
        out = self.activation_fn(out)

        if is_training:
            out = hk.dropout(kh.next_key(), 0.5, out)
        out = self.fc_2(out, kh.next_key(), stochastic)
        return out


class OmniglotCNN(hk.Module):
    """
    CNN intended to match ConvNetworkWithBias in the FRCL [1] repo. This architecture
    was previously used in [2] with inspiration from [3].

    [1] Titsias et al (2020). Functional regularisation for continual learning with Gaussian processes.
    [2] Schwarz et al (2018). Progress & compress: a scalable framework for continual learning.
    [3] Vinyals et al (2016). Matching networks for one shot learning.
    """

    def __init__(
        self,
        output_dim: int,
        activation_fn: Callable,
        final_layer_variational: bool,
        uniform_init_minval: float = -10,
        uniform_init_maxval: float = -8,
        channels: Sequence[int] = (64, 64, 64, 64),
    ):
        """
        :param output_dim: the number of output dimensions.
        :param activation_fn: activation function.
        :param final_layer_variational: if True, then all the layers except
            the last layer don't have variance for variational parameters.
        :param uniform_init_minval: lower bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param uniform_init_maxval: upper bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param channels: number of channels for each convolution layer.
        """
        super().__init__()
        self.activation_fn = activation_fn
        self.max_pool = hk.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
        )

        inner_layers_stochastic = not final_layer_variational

        self.conv = []
        for channels_i in channels:
            conv_i = Conv2DStochastic(
                output_channels=channels_i,
                kernel_shape=3,
                stochastic_parameters=inner_layers_stochastic,
                init_logvar_minval=uniform_init_minval,
                init_logvar_maxval=uniform_init_maxval,
            )
            self.conv.append(conv_i)

        self.fc = DenseStochastic(
            output_size=output_dim,
            stochastic_parameters=True,
            name="linear_final",
            init_logvar_minval=uniform_init_minval,
            init_logvar_maxval=uniform_init_maxval,
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
        del is_training
        kh = KeyHelper(rng_key)
        out = inputs
        for i in range(4):
            out = self.conv[i](out, kh.next_key(), stochastic)
            out = self.max_pool(out)
            out = self.activation_fn(out)

        out = jnp.moveaxis(out, source=3, destination=1)
        out = out.reshape(inputs.shape[0], -1)
        out = jnp.concatenate(
            (out, jnp.ones((out.shape[0], 1), dtype=out.dtype)), axis=1
        )
        out = self.fc(out, kh.next_key(), stochastic)
        return out
