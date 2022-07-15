from typing import Callable, Sequence

import haiku as hk
import jax.numpy as jnp

from haiku import BatchNorm as BatchNorm_reg
from sfsvi.models.haiku_mod import BatchNorm as BatchNorm_mod

from sfsvi.models.haiku_mod import conv2D_stochastic, dense_stochastic_hk
from sfsvi.general_utils.jax_utils import KeyHelper
from sfsvi.fsvi_utils.utils_cl import get_inner_layers_stochastic


class CNN(hk.Module):
    def __init__(
        self,
        output_dim: int,
        activation_fn: Callable,
        no_final_layer_bias: bool,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        batch_normalization: bool,
        batch_normalization_mod: bool,
        x_condition,
        final_layer_variational: bool,
        fixed_inner_layers_variational_var: bool,
        channels: Sequence[int],
        fc_dim: int,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.no_final_layer_bias = no_final_layer_bias
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition
        self.n_condition = self.x_condition.shape[0] if self.x_condition is not None else 0

        bn_config = {}
        bn_config["eps"] = 1e-5
        bn_config["decay_rate"] = 0.9

        if self.batch_normalization_mod != "not_specified":
            self.BatchNorm = BatchNorm_mod
            if self.batch_normalization_mod == "training_evaluation":
                bn_config["condition_mode"] = "training_evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
            elif self.batch_normalization_mod == "evaluation":
                bn_config["condition_mode"] = "evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
        else:
            self.BatchNorm = BatchNorm_reg
            bn_config["n_condition"] = self.n_condition
            bn_config["create_scale"] = True
            bn_config["create_offset"] = True

        self.max_pool = hk.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
        )

        inner_layers_stochastic = get_inner_layers_stochastic(
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
        )

        layer_args_lin = dict(
            uniform_init_minval=uniform_init_lin_minval,
            uniform_init_maxval=uniform_init_lin_maxval,
            # w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
            # b_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
        )

        layer_args_conv = dict(
            uniform_init_minval=uniform_init_conv_minval,
            uniform_init_maxval=uniform_init_conv_maxval,
            # w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
            # b_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
        )

        final_layer_args = dict(
            uniform_init_minval=uniform_init_lin_minval,
            uniform_init_maxval=uniform_init_lin_maxval,
            with_bias = not no_final_layer_bias,
            # w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
            # b_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "uniform"),
        )

        self.conv, self.batch_norm = [], []
        for channels_i in channels:
            conv_i = conv2D_stochastic(
                output_channels=channels_i,
                kernel_shape=3,
                stochastic_parameters=inner_layers_stochastic,
                **layer_args_conv,
            )
            if self.batch_normalization:
                batch_norm_i = self.BatchNorm(
                    name="batchnorm", **bn_config,
                )
            else:
                batch_norm_i = None
            self.conv.append(conv_i)
            self.batch_norm.append(batch_norm_i)

        self.fc_1 = dense_stochastic_hk(
            output_size=fc_dim,
            stochastic_parameters=inner_layers_stochastic,
            **layer_args_lin,
            # name="linear_penultimate",
        )
        self.fc_2 = dense_stochastic_hk(
            output_size=output_dim,
            stochastic_parameters=stochastic_parameters,
            name="linear_final",
            **final_layer_args,
        )


class SixLayers(CNN):
    """CNN with 4 conv layers and 2 fc layers."""
    def __init__(self, *args, **kwargs):
        # https://www.arxiv-vanity.com/papers/1703.04200/
        super().__init__(channels=(32, 32, 64, 64), fc_dim=512, *args, **kwargs)

    def __call__(self, inputs, rng_key, stochastic, is_training):
        if self.n_condition > 0:
            inputs = jnp.concatenate([inputs, self.x_condition], axis=0)

        # TODO: remove hardcoded dropout rates.
        kh = KeyHelper(rng_key)
        out = inputs

        for i in range(2):
            out = self.conv[i](out, kh.next_key(), stochastic)
            if self.batch_norm[i] != None:
                out = self.batch_norm[i](out, is_training=is_training)
            out = self.activation_fn(out)
        out = self.max_pool(out)

        if is_training:
            out = hk.dropout(kh.next_key(), 0.25, out)

        for i in range(2, 4):
            out = self.conv[i](out, kh.next_key(), stochastic)
            if self.batch_norm[i] != None:
                out = self.batch_norm[i](out, is_training=is_training)
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

        if self.n_condition > 0:
            out = out[:-self.n_condition]

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
        stochastic_parameters: bool,
        no_final_layer_bias: bool,
        dropout: bool,
        dropout_rate: float,
        batch_normalization: bool,
        batch_normalization_mod: bool,
        final_layer_variational: bool,
        fixed_inner_layers_variational_var: bool,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
        channels: Sequence[int] = (64, 64, 64, 64),
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.max_pool = hk.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
        )

        inner_layers_stochastic = get_inner_layers_stochastic(
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
        )

        layer_args_lin = dict(
            uniform_init_minval=uniform_init_lin_minval,
            uniform_init_maxval=uniform_init_lin_maxval,
        )

        layer_args_conv = dict(
            uniform_init_minval=uniform_init_conv_minval,
            uniform_init_maxval=uniform_init_conv_maxval,
        )

        self.conv = []
        for channels_i in channels:
            conv_i = conv2D_stochastic(
                output_channels=channels_i,
                kernel_shape=3,
                stochastic_parameters=inner_layers_stochastic,
                **layer_args_conv,
            )
            self.conv.append(conv_i)

        self.fc = dense_stochastic_hk(
            output_size=output_dim,
            stochastic_parameters=stochastic_parameters,
            name="linear_final",
            **layer_args_lin,
        )

    def __call__(self, inputs, rng_key, stochastic, is_training):
        del is_training
        kh = KeyHelper(rng_key)
        out = inputs
        for i in range(4):
            out = self.conv[i](out, kh.next_key(), stochastic)
            out = self.max_pool(out)
            out = self.activation_fn(out)

        out = jnp.moveaxis(out, source=3, destination=1)
        out = out.reshape(inputs.shape[0], -1)
        out = jnp.concatenate((out, jnp.ones((out.shape[0], 1), dtype=out.dtype)), axis=1)
        out = self.fc(out, kh.next_key(), stochastic)
        return out
