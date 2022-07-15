from typing import Callable

import haiku as hk
import jax.numpy as jnp

from haiku import BatchNorm as BatchNorm_reg
from sfsvi.models.haiku_mod import BatchNorm as BatchNorm_mod

from sfsvi.models.haiku_mod import dense_stochastic_hk
from sfsvi.general_utils.jax_utils import KeyHelper
from sfsvi.fsvi_utils.utils_cl import get_inner_layers_stochastic


class MLP(hk.Module):
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
        network_type: str,
        architecture,
        init_logvar_lin_minval: float,
        init_logvar_lin_maxval: float,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.output_dim = output_dim
        self.architecture = architecture
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

        inner_layers_stochastic = get_inner_layers_stochastic(
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
        )

        if network_type == "fully_connected":
            self.layers = []
            self.batch_norm_layers = []
            for unit in architecture[:-1]:
                self.layers.append(
                    dense_stochastic_hk(
                        unit,
                        uniform_init_minval=init_logvar_lin_minval,
                        uniform_init_maxval=init_logvar_lin_maxval,
                        stochastic_parameters=inner_layers_stochastic,
                    )
                )
                if self.batch_normalization:
                    self.batch_norm_layers.append(
                        self.BatchNorm(
                            name="batchnorm", **bn_config,
                        )
                    )
                else:
                    self.batch_norm_layers.append(None)
        elif network_type == "resnet":
            # FIRST LAYER
            self.layers = [
                dense_stochastic_hk(
                    architecture[0],
                    uniform_init_minval=init_logvar_lin_minval,
                    uniform_init_maxval=init_logvar_lin_maxval,
                    stochastic_parameters=inner_layers_stochastic,
                )
            ]
            if self.batch_normalization:
                self.batch_norm_layers.append(
                    self.BatchNorm(
                        name="batchnorm", **bn_config,
                    )
                )
            else:
                self.batch_norm_layers.append(None)

            # RESIDUAL LAYERS
            for unit in architecture[1:-1]:
                self.layers.append(
                    dense_stochastic_hk(
                        unit,
                        uniform_init_minval=init_logvar_lin_minval,
                        uniform_init_maxval=init_logvar_lin_maxval,
                        stochastic_parameters=inner_layers_stochastic,
                    )
                )
                if self.batch_normalization:
                    self.batch_norm_layers.append(
                        self.BatchNorm(
                            name="batchnorm", **bn_config,
                        )
                    )
                else:
                    self.batch_norm_layers.append(None)

        else:
            raise ValueError("Network type not defined.")

        # PENULTIMATE LAYER
        self.layers.append(
            dense_stochastic_hk(
                architecture[-1],
                uniform_init_minval=init_logvar_lin_minval,
                uniform_init_maxval=init_logvar_lin_maxval,
                stochastic_parameters=inner_layers_stochastic,
            )
        )
        if self.batch_normalization:
            self.batch_norm_layers.append(
                self.BatchNorm(
                    name="batchnorm", **bn_config,
                )
            )
        else:
            self.batch_norm_layers.append(None)

        # FINAL LAYER
        self.layers.append(
            dense_stochastic_hk(
                self.output_dim,
                uniform_init_minval=init_logvar_lin_minval,
                uniform_init_maxval=init_logvar_lin_maxval,
                stochastic_parameters=stochastic_parameters,
                with_bias= not self.no_final_layer_bias,
                name="linear_final",
            )
        )


class FullyConnected(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(network_type="fully_connected", *args, **kwargs)

    def __call__(self, inputs, rng_key, stochastic, is_training):
        if self.n_condition > 0:
            inputs = jnp.concatenate([inputs, self.x_condition], axis=0)

        kh = KeyHelper(rng_key)
        out = inputs

        for l in range(len(self.layers)-2):
            out = self.layers[l](out, kh.next_key(), stochastic)
            if self.batch_norm_layers[l] != None:
                out = self.batch_norm_layers[l](out, is_training=is_training)
            out = self.activation_fn(out)

        # PENULTIMATE LAYER
        out = self.activation_fn(self.layers[-2](out, kh.next_key(), stochastic))

        # FINAL LAYER
        out = self.layers[-1](out, rng_key, stochastic)

        if self.n_condition > 0:
            out = out[:-self.n_condition]

        return out
