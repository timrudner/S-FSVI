from functools import partial
from typing import Callable
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit

from sfsvi.models import custom_cnns
from sfsvi.models import custom_mlps
from sfsvi.models.haiku_mod import gaussian_sample_pytree
from sfsvi.models.haiku_mod import partition_all_params
from sfsvi.models.haiku_mod import partition_params_final_layer_bnn
from sfsvi.models.haiku_mod import predicate_batchnorm
from sfsvi.models.haiku_mod import predicate_mean
from sfsvi.fsvi_utils.utils_linearization import convert_predict_f_only_mean

relu = jax.nn.relu
tanh = jnp.tanh

eps = 1e-6

ACTIVATION_DICT = {"tanh": jnp.tanh, "relu": jax.nn.relu, "lrelu": jax.nn.leaky_relu, "elu": jax.nn.elu}


class Model:
    def __init__(
        self,
        output_dim: int,
        architecture: str,
        no_final_layer_bias: bool,
        activation_fn: str = "relu",
        stochastic_parameters: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        extra_linear_layer: bool = False,
        feature_map_jacobian: bool = False,
        feature_map_jacobian_train_only: bool = False,
        feature_map_type: str = "not_specified",
        regularization=0.0,
        dropout=False,
        dropout_rate=0.0,
        resnet=False,
        batch_normalization=False,
        batch_normalization_mod="not_specified",
        x_condition=None,
        init_logvar_minval=0.0,
        init_logvar_maxval=0.0,
        init_logvar_lin_minval=0.0,
        init_logvar_lin_maxval=0.0,
        init_logvar_conv_minval=0.0,
        init_logvar_conv_maxval=0.0,
        perturbation_param=0.01,
    ):
        """

        @param stochastic_parameters:
        @param final_layer_variational: if True, then all the parameters except the last layer are set to be deterministic.
        """
        self.output_dim = output_dim
        self.regularization = regularization
        self.final_layer_variational = final_layer_variational
        self.fixed_inner_layers_variational_var = fixed_inner_layers_variational_var
        self.extra_linear_layer = extra_linear_layer
        self.feature_map_jacobian = feature_map_jacobian
        self.feature_map_jacobian_train_only = feature_map_jacobian_train_only
        self.feature_map_type = feature_map_type
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.architecture = architecture
        self.no_final_layer_bias = no_final_layer_bias
        self.stochastic_parameters = stochastic_parameters
        self.resnet = resnet
        self.init_logvar_minval = init_logvar_minval
        self.init_logvar_maxval = init_logvar_maxval
        self.init_logvar_lin_minval = init_logvar_lin_minval
        self.init_logvar_lin_maxval = init_logvar_lin_maxval
        self.init_logvar_conv_minval = init_logvar_conv_minval
        self.init_logvar_conv_maxval = init_logvar_conv_maxval
        self.perturbation_param = perturbation_param

        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition
        self.n_condition = self.x_condition.shape[0] if self.x_condition is not None else 0
        self.params_init = None
        self.params_eval_carry = None

        self.inner_layers_stochastic = get_inner_layers_stochastic(
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var
        )

        if self.architecture == "vit":
            self.forward = self.make_forward_fn()()
        else:
            self.forward = hk.transform_with_state(self.make_forward_fn())

    @property
    def apply_fn(self):
        return self.forward.apply

    def make_forward_fn(self):
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 6,))
    def predict_f(self, params, params_feature, state, inputs, rng_key, is_training):
        return self._predict_f(params, params_feature, state, inputs, rng_key, is_training)

    def _predict_f(self, params, params_feature, state, inputs, rng_key, is_training):
        if self.feature_map_jacobian:
            def _predict_f(params, params_feature, state, rng_key, inputs, stochastic, is_training):
                if not self.final_layer_variational:
                    params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
                    params_feature_mean, params_feature_batchnorm, params_feature_log_var, params_feature_rest = partition_all_params(params_feature)

                    stochastic_model = self.stochastic_parameters and self.inner_layers_stochastic
                    if stochastic_model:
                        params_lin = gaussian_sample_pytree(params_mean, params_log_var, rng_key)
                    else:
                        perturbation_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * self.perturbation_param) ** 2 ), jax.lax.stop_gradient(params_feature_mean)))
                        params_lin = gaussian_sample_pytree(params_mean, perturbation_log_var, rng_key)

                    # perturbation_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * 0.01) ** 2 ), _params_mean))
                    #
                    # sample = self.stochastic_parameters and self.inner_layers_stochastic
                    # if sample:
                    #     params_mean = gaussian_sample_pytree(_params_mean, params_log_var, rng_key)
                    #     params_feature_mean = gaussian_sample_pytree(_params_feature_mean, params_log_var, rng_key)
                    #     params_lin = gaussian_sample_pytree(params_mean, perturbation_log_var, rng_key)
                    # else:
                    #     params_lin = gaussian_sample_pytree(_params_mean, perturbation_log_var, rng_key)
                else:
                    # checks if params == params_feature, since "else" is only defined for this case
                    # assert jax.tree_multimap(lambda x, y: x - y, params, params_feature) == 0
                    params_mean_final_layer, params_log_var_final_layer, params_penultimate = partition_params_final_layer_bnn(params)
                    params_feature_batchnorm = hk.data_structures.filter(predicate_batchnorm, params)
                    params_feature_rest = hk.data_structures.merge(params_log_var_final_layer, params_penultimate)

                    params_mean = params_feature_mean = params_mean_final_layer
                    params_lin = gaussian_sample_pytree(params_mean_final_layer, params_log_var_final_layer, rng_key)

                if self.feature_map_type == "learned_grad":
                    params_eval = params_mean
                elif self.feature_map_type == "learned_nograd":
                    params_eval = jax.lax.stop_gradient(params_feature_mean)
                elif self.feature_map_type == "init":
                    params_eval = hk.data_structures.filter(predicate_mean, self.params_init)
                # elif self.feature_map_type == "learned_nograd":
                #     params_eval = hk.data_structures.filter(predicate_mean, params_featurehk.data_structures.filter(predicate_mean, self.params_init))
                else:
                    raise ValueError("Jacobian feature map not specified.")

                _pred_fn = lambda _params, rng_key: convert_predict_f_only_mean(  # returns a function of params_mean
                    self.apply_fn, inputs, params_feature_rest, params_feature_batchnorm, state, rng_key, stochastic_model, is_training
                )(_params)

                # _params_lin = params_lin
                _params_lin = jax.tree_multimap(lambda x, y: x - y, params_lin, params_eval)

                if is_training:
                    pred_f, pred_jvp = jax.jvp(partial(_pred_fn, rng_key=rng_key), (params_eval,), (_params_lin,))
                else:
                    if self.feature_map_jacobian_train_only:
                        pred_f = _pred_fn(_params=params_lin, rng_key=rng_key)
                    else:
                        pred_f, pred_jvp = jax.jvp(partial(_pred_fn, rng_key=rng_key), (params_eval,), (_params_lin,))

                ## vmaped version of predictive function (does not scale well with the minibatch size)
                # def pred_fn_vmap(inputs):
                #     _pred_fn = lambda _params, _inputs: convert_predict_f_only_mean(  # returns a function of params_mean
                #         self.apply_fn, _inputs, params_rest, params_batchnorm, state, rng_key, stochastic, is_training
                #     )(_params)
                #     _pred_batch_fn = lambda _inputs: jax.jvp(partial(_pred_fn, _inputs=_inputs), (params_eval,), (params_lin,))
                #     _pred_fn_vmap = jax.vmap(_pred_batch_fn, in_axes=0, out_axes=0)
                #     inputs_expanded = jnp.expand_dims(inputs, axis=1)
                #     pred_f_expanded, pred_jvp_expanded = _pred_fn_vmap(inputs_expanded)
                #     pred_f, pred_jvp = jnp.squeeze(pred_f_expanded, axis=1), jnp.squeeze(pred_jvp_expanded, axis=1)
                #
                #     return pred_f, pred_jvp
                #
                # pred_f, pred_jvp = jax.jit(pred_fn_vmap)(inputs)

                if is_training:
                    pred = pred_f + pred_jvp
                else:
                    if self.feature_map_jacobian_train_only:
                        pred = pred_f
                    else:
                        pred = pred_f + pred_jvp

                return pred

                ## Check accuracy of linearization
                # params_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * 0.01) ** 2 ), params_mean))
                # params_lin = jax.lax.stop_gradient(gaussian_sample_pytree(params_mean, params_log_var, rng_key))
                #
                # x = jax.lax.stop_gradient(_pred_fn(params_mean, rng_key=rng_key))
                # y = jax.lax.stop_gradient(_pred_fn(params_lin, rng_key=rng_key))
                #
                # z = x - y

            return _predict_f(
                params,
                params_feature,
                state,
                rng_key,
                inputs,
                True,
                is_training,
            )
        else:
            # does not use params_feature:
            return self.forward.apply(
                params,
                state,
                rng_key,
                inputs,
                rng_key,
                stochastic=True,
                is_training=is_training,
            )[0]

    @partial(jit, static_argnums=(0,6,))
    def predict_f_deterministic(self, params, params_feature, state, inputs, rng_key, is_training):
        return self._predict_f_deterministic(params, params_feature, state, inputs, rng_key, is_training)

    def _predict_f_deterministic(self, params, params_feature, state, inputs, rng_key, is_training):
        """
        Forward pass with mean parameters (hence deterministic)
        """
        if self.feature_map_jacobian:
            def _predict_f(params, params_feature, state, rng_key, inputs, stochastic, is_training):
                if not self.final_layer_variational:
                    params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
                    params_feature_mean, params_feature_batchnorm, params_feature_log_var, params_feature_rest = partition_all_params(params_feature)

                    perturbation_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * self.perturbation_param) ** 2 ), jax.lax.stop_gradient(params_feature_mean)))
                    params_lin = gaussian_sample_pytree(params_mean, perturbation_log_var, rng_key)
                else:
                    # checks if params == params_feature, since "else" is only defined for this case
                    # assert jax.tree_multimap(lambda x, y: x - y, params, params_feature) == 0
                    params_mean_final_layer, params_log_var_final_layer, params_penultimate = partition_params_final_layer_bnn(params)
                    params_feature_batchnorm = hk.data_structures.filter(predicate_batchnorm, params)
                    params_feature_rest = hk.data_structures.merge(params_log_var_final_layer, params_penultimate)

                    params_mean = params_feature_mean = params_mean_final_layer
                    params_lin = gaussian_sample_pytree(params_mean_final_layer, params_log_var_final_layer, rng_key)

                if self.feature_map_type == "learned_grad":
                    params_eval = params_mean
                elif self.feature_map_type == "learned_nograd":
                    params_eval = jax.lax.stop_gradient(params_feature_mean)
                elif self.feature_map_type == "init":
                    params_eval = hk.data_structures.filter(predicate_mean, self.params_init)
                # elif self.feature_map_type == "learned_nograd":
                #     params_eval = hk.data_structures.filter(predicate_mean, params_featurehk.data_structures.filter(predicate_mean, self.params_init))
                else:
                    raise ValueError("Jacobian feature map not specified.")

                _pred_fn = lambda _params, rng_key: convert_predict_f_only_mean(  # returns a function of params_mean
                    self.apply_fn, inputs, params_feature_rest, params_feature_batchnorm, state, rng_key, stochastic, is_training
                )(_params)

                # _params_lin = params_lin
                _params_lin = jax.tree_multimap(lambda x, y: x - y, params_lin, params_eval)

                pred_f, pred_jvp = jax.jvp(partial(_pred_fn, rng_key=rng_key), (params_eval,), (_params_lin,))

                if is_training:
                    pred = pred_f + pred_jvp
                else:
                    if self.feature_map_jacobian_train_only:
                        pred = pred_f
                    else:
                        pred = pred_f + pred_jvp

                return pred

                ## Check accuracy of linearization
                # params_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * 0.01) ** 2 ), params_mean))
                # params_lin = jax.lax.stop_gradient(gaussian_sample_pytree(params_mean, params_log_var, rng_key))
                #
                # x = jax.lax.stop_gradient(_pred_fn(params_mean, rng_key=rng_key))
                # y = jax.lax.stop_gradient(_pred_fn(params_lin, rng_key=rng_key))
                #
                # z = x - y

            return _predict_f(
                params,
                params_feature,
                state,
                rng_key,
                inputs,
                True,
                is_training,
            )
        else:
            # does not use params_feature:
            return self.forward.apply(
                params,
                state,
                rng_key,
                inputs,
                rng_key,
                stochastic=False,
                is_training=is_training,
            )[0]

        # return self.forward.apply(
        #     params,
        #     state,
        #     rng_key,
        #     inputs,
        #     rng_key,
        #     stochastic=False,
        #     is_training=is_training,
        # )[0]

    @partial(jit, static_argnums=(0, 5,))
    def predict_y(self, params, state, inputs, rng_key, is_training):
        return jax.nn.softmax(
            self.predict_f(params, params, state, inputs, rng_key, is_training)
        )

    def predict_f_multisample(
        self, params, params_feature, state, inputs, rng_key, n_samples: int, is_training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        @return:
            preds_samples: an array of shape (n_samples, inputs.shape[0], output_dimension)
            preds_mean: an array of shape (inputs.shape[0], output_dimension)
            preds_var: an array of shape (inputs.shape[0], output_dimension)
        """
        # TODO: test if vmap can accelerate this code
        pred_fn = lambda rng_key: self.predict_f(params, params_feature, state, inputs, rng_key, is_training)

        samples = mc_sampling(
            fn=pred_fn,
            n_samples=n_samples,
            rng_key=rng_key,
        )

        return samples

    def predict_y_multisample(
        self, params, state, inputs, rng_key, n_samples, is_training
    ):
        return mc_sampling(
            fn=lambda _rng_key: self.predict_y(params, state, inputs, _rng_key, is_training),
            n_samples=n_samples,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0,6,7,))
    def predict_f_multisample_jitted(
        self, params, params_feature, state, inputs, rng_key, n_samples: int, is_training: bool,
    ):
        """
        This is jitted version of predict_f_multisample
        """
        ### vmap
        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_f(
            params, params_feature, state, inputs, rng_key, is_training,
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )  # fastest for n_samples=10
        preds_samples = predict_multisample_fn(rng_keys)

        preds_mean = preds_samples.mean(axis=0)
        preds_var = preds_samples.std(axis=0) ** 2
        return preds_samples, preds_mean, preds_var

    @partial(jit, static_argnums=(0, 5, 6,))
    def predict_y_multisample_jitted(
        self, params, state, inputs, rng_key, n_samples, is_training
    ):
        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_y(
            params, state, inputs, rng_key, is_training
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )
        preds_samples = predict_multisample_fn(rng_keys)
        preds_mean = preds_samples.mean(0)
        preds_var = preds_samples.std(0) ** 2
        return preds_samples, preds_mean, preds_var


def get_inner_layers_stochastic(
    stochastic_parameters: bool,
    final_layer_variational: bool,
    fixed_inner_layers_variational_var: bool,
):
    if stochastic_parameters:
        inner_layers_stochastic = not final_layer_variational or fixed_inner_layers_variational_var
    else:
        inner_layers_stochastic = False
    return inner_layers_stochastic


class MLP(Model):
    def __init__(
        self,
        output_dim: int,
        architecture: str,
        no_final_layer_bias: bool,
        activation_fn: str = "relu",
        stochastic_parameters: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        extra_linear_layer: bool = False,
        feature_map_jacobian: bool = False,
        feature_map_jacobian_train_only: bool = False,
        feature_map_type: str = "not_specified",
        regularization=0.0,
        dropout=False,
        dropout_rate=0.0,
        resnet=False,
        batch_normalization=False,
        batch_normalization_mod="not_specified",
        x_condition=None,
        init_logvar_minval=0.0,
        init_logvar_maxval=0.0,
        init_logvar_lin_minval=0.0,
        init_logvar_lin_maxval=0.0,
        init_logvar_conv_minval=0.0,
        init_logvar_conv_maxval=0.0,
        perturbation_param=0.01,
    ):
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition

        super().__init__(
            output_dim=output_dim,
            architecture=architecture,
            no_final_layer_bias=no_final_layer_bias,
            activation_fn=activation_fn,
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
            extra_linear_layer=extra_linear_layer,
            feature_map_jacobian=feature_map_jacobian,
            feature_map_jacobian_train_only=feature_map_jacobian_train_only,
            feature_map_type=feature_map_type,
            regularization=regularization,
            dropout=dropout,
            dropout_rate=dropout_rate,
            init_logvar_minval=init_logvar_minval,
            init_logvar_maxval=init_logvar_maxval,
            init_logvar_lin_minval=init_logvar_lin_minval,
            init_logvar_lin_maxval=init_logvar_lin_maxval,
            perturbation_param=perturbation_param,
            batch_normalization=self.batch_normalization,
            batch_normalization_mod=self.batch_normalization_mod,
            x_condition=self.x_condition,
        )

    def make_forward_fn(self):
        # TODO: Maybe remove hardcoding here
        if self.init_logvar_minval == 0.0 and self.init_logvar_lin_minval == 0.0 and self.init_logvar_conv_minval == 0.0:
            self.init_logvar_lin_minval = -10.0
            self.init_logvar_lin_maxval = -8.0
        if self.init_logvar_lin_minval == 0.0 and self.init_logvar_minval < 0.0:
            self.init_logvar_lin_minval = self.init_logvar_minval
            self.init_logvar_lin_maxval = self.init_logvar_maxval

        print(f"init_logvar_lin_minval: {self.init_logvar_lin_minval}")
        print(f"init_logvar_lin_maxval: {self.init_logvar_lin_maxval}")

        def forward_fn(inputs, rng_key, stochastic, is_training):
            _forward_fn = custom_mlps.FullyConnected(
                output_dim=self.output_dim,
                activation_fn=self.activation_fn,
                architecture=self.architecture,
                no_final_layer_bias=self.no_final_layer_bias,
                stochastic_parameters=self.stochastic_parameters,
                dropout=self.dropout,
                dropout_rate=self.dropout_rate,
                batch_normalization=self.batch_normalization,
                batch_normalization_mod=self.batch_normalization_mod,
                x_condition=self.x_condition,
                final_layer_variational=self.final_layer_variational,
                fixed_inner_layers_variational_var=self.fixed_inner_layers_variational_var,
                init_logvar_lin_minval=self.init_logvar_lin_minval,
                init_logvar_lin_maxval=self.init_logvar_lin_maxval,
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
        stochastic_parameters: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        extra_linear_layer: bool = False,
        feature_map_jacobian: bool = False,
        feature_map_jacobian_train_only: bool = False,
        feature_map_type: str = "not_specified",
        regularization=0.0,
        dropout=False,
        dropout_rate=0.0,
        resnet=False,
        batch_normalization=False,
        batch_normalization_mod="not_specified",
        x_condition=None,
        init_logvar_minval=0.0,
        init_logvar_maxval=0.0,
        init_logvar_lin_minval=0.0,
        init_logvar_lin_maxval=0.0,
        init_logvar_conv_minval=0.0,
        init_logvar_conv_maxval=0.0,
        perturbation_param=0.01,
    ):
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition

        super().__init__(
            output_dim=output_dim,
            architecture=architecture,
            no_final_layer_bias=no_final_layer_bias,
            activation_fn=activation_fn,
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
            extra_linear_layer=extra_linear_layer,
            feature_map_jacobian=feature_map_jacobian,
            feature_map_jacobian_train_only=feature_map_jacobian_train_only,
            feature_map_type=feature_map_type,
            regularization=regularization,
            dropout=dropout,
            dropout_rate=dropout_rate,
            init_logvar_minval=init_logvar_minval,
            init_logvar_maxval=init_logvar_maxval,
            init_logvar_lin_minval=init_logvar_lin_minval,
            init_logvar_lin_maxval=init_logvar_lin_maxval,
            init_logvar_conv_minval=init_logvar_conv_minval,
            init_logvar_conv_maxval=init_logvar_conv_maxval,
            perturbation_param=perturbation_param,
            batch_normalization=self.batch_normalization,
            batch_normalization_mod=self.batch_normalization_mod,
            x_condition=self.x_condition,
        )

    def make_forward_fn(self):
        # TODO: Maybe remove hardcoding here
        if self.init_logvar_minval == 0.0 and self.init_logvar_lin_minval == 0.0 and self.init_logvar_conv_minval == 0.0:
            self.init_logvar_lin_minval = -10.0
            self.init_logvar_lin_maxval = -8.0
            if "resnet" in self.architecture:
                self.init_logvar_lin_minval = -20.0
                self.init_logvar_lin_maxval = -18.0
        if self.init_logvar_minval == 0.0 and self.init_logvar_conv_minval == 0.0:
            self.init_logvar_conv_minval = -10.0
            self.init_logvar_conv_maxval = -8.0
            if "resnet" in self.architecture:
                self.init_logvar_conv_minval = -20.0
                self.init_logvar_conv_maxval = -18.0
        if self.init_logvar_lin_minval == 0.0 and self.init_logvar_minval < 0.0:
            self.init_logvar_lin_minval = self.init_logvar_minval
            self.init_logvar_lin_maxval = self.init_logvar_maxval
        if self.init_logvar_conv_minval == 0.0 and self.init_logvar_minval < 0.0:
            self.init_logvar_conv_minval = self.init_logvar_minval
            self.init_logvar_conv_maxval = self.init_logvar_maxval

        print(f"init_logvar_lin_minval: {self.init_logvar_lin_minval}")
        print(f"init_logvar_lin_maxval: {self.init_logvar_lin_maxval}")
        print(f"init_logvar_conv_minval: {self.init_logvar_conv_minval}")
        print(f"init_logvar_conv_maxval: {self.init_logvar_conv_maxval}")

        if self.architecture == "six_layers":

            def forward_fn(inputs, rng_key, stochastic, is_training):
                _forward_fn = custom_cnns.SixLayers(
                    output_dim=self.output_dim,
                    activation_fn=self.activation_fn,
                    no_final_layer_bias=self.no_final_layer_bias,
                    stochastic_parameters=self.stochastic_parameters,
                    dropout=self.dropout,
                    dropout_rate=self.dropout_rate,
                    batch_normalization=self.batch_normalization,
                    batch_normalization_mod=self.batch_normalization_mod,
                    x_condition=self.x_condition,
                    final_layer_variational=self.final_layer_variational,
                    fixed_inner_layers_variational_var=self.fixed_inner_layers_variational_var,
                    uniform_init_lin_minval=self.init_logvar_lin_minval,
                    uniform_init_lin_maxval=self.init_logvar_lin_maxval,
                    uniform_init_conv_minval=self.init_logvar_conv_minval,
                    uniform_init_conv_maxval=self.init_logvar_conv_maxval,
                )
                return _forward_fn(inputs, rng_key, stochastic, is_training)

        elif self.architecture == "omniglot_cnn":

            def forward_fn(inputs, rng_key, stochastic, is_training):
                _forward_fn = custom_cnns.OmniglotCNN(
                    output_dim=self.output_dim,
                    activation_fn=self.activation_fn,
                    no_final_layer_bias=self.no_final_layer_bias,
                    stochastic_parameters=self.stochastic_parameters,
                    dropout=self.dropout,
                    dropout_rate=self.dropout_rate,
                    batch_normalization=self.batch_normalization,
                    batch_normalization_mod=self.batch_normalization_mod,
                    final_layer_variational=self.final_layer_variational,
                    fixed_inner_layers_variational_var=self.fixed_inner_layers_variational_var,
                    uniform_init_lin_minval=self.init_logvar_lin_minval,
                    uniform_init_lin_maxval=self.init_logvar_lin_maxval,
                    uniform_init_conv_minval=self.init_logvar_conv_minval,
                    uniform_init_conv_maxval=self.init_logvar_conv_maxval,
                )
                return _forward_fn(inputs, rng_key, stochastic, is_training)
        else:
            raise NotImplementedError(self.architecture)
        return forward_fn


def mc_sampling(
    fn: Callable, n_samples: int, rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs Monte Carlo sampling and returns the samples, the mean of samples and the variance of samples

    @param fn: a deterministic function that takes in a random key and returns one MC sample
    @param n_samples: number of MC samples
    @param rng_key: random key
    @return:
            preds_samples: an array of shape (n_samples, ) + `output_shape`, where `output_shape` is the shape
                of output of `fn`
            preds_mean: an array of shape `output_shape`
            preds_var: an array of shape `output_shape`
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
