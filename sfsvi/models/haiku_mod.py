# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import moving_averages
from haiku._src import utils
from jax import jit
from jax import lax
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd

hk.get_parameter = base.get_parameter
hk.initializers = initializers
hk.Module = module.Module
hk.ExponentialMovingAverage = moving_averages.ExponentialMovingAverage
del base, initializers, module, moving_averages

from sfsvi.general_utils.haiku_utils import map_variable_name

dtype_default = jnp.float32


# TODO: remove hard-coding of interval bounds
def uniform_mod(min_val, max_val):
    def _uniform_mod(shape, dtype):
        rng_key, _ = random.split(random.PRNGKey(0))
        return jax.random.uniform(
            rng_key, shape=shape, dtype=dtype, minval=min_val, maxval=max_val
            # rng_key, shape=shape, dtype=dtype, minval=-10.0, maxval=-8.0
            # rng_key, shape=shape, dtype=dtype, minval=-20.0, maxval=-18.0
            # rng_key, shape=shape, dtype=dtype, minval=-5.0, maxval=-4.0
            # rng_key, shape=shape, dtype=dtype, minval=-3.0, maxval=-2.5
            # rng_key, shape=shape, dtype=dtype, minval=-3.0, maxval=-2.0
        )
    return _uniform_mod


def gaussian_sample(mu: jnp.ndarray, rho: jnp.ndarray, stochastic: bool, rng_key):
    if stochastic:
        ## Gaussian variational distribution
        jnp_eps = random.normal(rng_key, mu.shape)
        z = mu + jnp.exp((0.5 * rho).astype(dtype_default)) * jnp_eps
        ## Laplace variational distribution
        # dist = tfd.Laplace(loc=mu, scale=jnp.exp(rho))
        # z = dist.sample(seed=rng_key)
    else:
        z = mu
    return z


def gaussian_sample_pytree(mu: jnp.ndarray, rho: jnp.ndarray, rng_key):
    rho_renamed = map_variable_name(rho, lambda n: f"{n.split('_')[0]}_mu")
    jnp_eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), rho_renamed)
    sigma = jax.tree_map(lambda x: jnp.exp((0.5 * x)), rho_renamed)
    sigma_eps = jax.tree_multimap(lambda x, y: x * y, sigma, jnp_eps)
    z = jax.tree_multimap(lambda x, y: x + y, mu, sigma_eps)
    return z


def multivariate_gaussian_sample(mu: jnp.ndarray, rho: jnp.ndarray, stochastic: bool, rng_key):
    if stochastic:
        ## Multivariate Gaussian variational distribution
        cov = jnp.exp(rho)
        if len(cov.shape) == 2:
            cov_psd = jnp.matmul(cov, cov.transpose())
            cov_psd += jnp.eye(cov_psd.shape[-1]) * 1e-7
        elif len(cov.shape) == 3:
            cov_psd = jnp.matmul(cov, cov.transpose([0,2,1]))
            cov_psd += jnp.repeat(jnp.expand_dims(jnp.eye(cov_psd.shape[-1]) * 1e-7, 0), repeats=cov_psd.shape[0], axis=0)
        else:
            raise NotImplementedError
        dist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov_psd)
        z = dist.sample(seed=rng_key)
    else:
        z = mu
    return z


def predicate_mean(module_name, name, value):
    return name == "w_mu" or name == "b_mu"


def predicate_var(module_name, name, value):
    return name == "w_logvar" or name == "b_logvar"


def predicate_batchnorm(module_name, name, value):
    return name not in {
        "w_mu",
        "b_mu",
        "w_logvar",
        "b_logvar",
    }


class dense_stochastic_hk(hk.Module):
    def __init__(
        self,
        output_size: int,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        stochastic_parameters: bool = False,
    ):
        super(dense_stochastic_hk, self).__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        self.stochastic_parameters = stochastic_parameters

    def __call__(self, inputs, rng_key, stochastic: bool):
        """
        @param stochastic: if True, use sampled parameters, otherwise, use mean parameters.
        @return:
        """
        j, k = inputs.shape[-1], self.output_size
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        if self.w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            self.w_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
        w_mu = hk.get_parameter("w_mu", shape=[j, k], dtype=dtype, init=self.w_init)

        if self.with_bias:
            if self.b_init is None:
                stddev = 1.0 / np.sqrt(self.input_size)
                self.b_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
            b_mu = hk.get_parameter("b_mu", shape=[k], dtype=dtype, init=self.b_init)
        if self.stochastic_parameters:
            if "XXX" in self.name:  # this is temporary; set to "final" for multivariate normal final layer variational distribution
                w_logvar = hk.get_parameter(
                    "w_logvar", shape=[j, k, 3], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                    # "w_logvar", shape=[j, k, 3], dtype=dtype, init=hk.initializers.RandomUniform(minval=-jnp.log(stddev), maxval=jnp.log(stddev))
                )
                if self.with_bias:
                    b_logvar = hk.get_parameter(
                        "b_logvar", shape=[k, 3], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                        # "b_logvar", shape=[k, 3], dtype=dtype, init=hk.initializers.RandomUniform(minval=-jnp.log(stddev), maxval=jnp.log(stddev))
                    )
                key_1, key_2 = jax.random.split(rng_key)
                W = multivariate_gaussian_sample(w_mu, w_logvar, stochastic, key_1)
                if self.with_bias:
                    b = multivariate_gaussian_sample(b_mu, b_logvar, stochastic, key_2)
                    return jnp.dot(inputs, W) + b
                else:
                    return jnp.dot(inputs, W)
            else:
                w_logvar = hk.get_parameter(
                    "w_logvar", shape=[j, k], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                    # "w_logvar", shape=[j, k, 3], dtype=dtype, init=hk.initializers.RandomUniform(minval=-jnp.log(stddev), maxval=jnp.log(stddev))
                )
                if self.with_bias:
                    b_logvar = hk.get_parameter(
                        "b_logvar", shape=[k], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                        # "b_logvar", shape=[k, 3], dtype=dtype, init=hk.initializers.RandomUniform(minval=-jnp.log(stddev), maxval=jnp.log(stddev))
                    )
                key_1, key_2 = jax.random.split(rng_key)
                W = gaussian_sample(w_mu, w_logvar, stochastic, key_1)
                if self.with_bias:
                    b = gaussian_sample(b_mu, b_logvar, stochastic, key_2)
                    return jnp.dot(inputs, W) + b
                else:
                    return jnp.dot(inputs, W)
        else:
            if self.with_bias:
                return jnp.dot(inputs, w_mu) + b_mu
            else:
                return jnp.dot(inputs, w_mu)


class conv2D_stochastic(hk.Module):
    """General N-dimensional convolutional."""

    def __init__(
        self,
        output_channels: int,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        kernel_shape: Union[int, Sequence[int]],
        num_spatial_dims: int = 2,
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[
            str, Sequence[Tuple[int, int]], hk.pad.PadFn, Sequence[hk.pad.PadFn]
        ] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: str = "channels_last",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        name: Optional[str] = None,
        stochastic_parameters: bool = False,
    ):
        """Initializes the module.
        Args:
            num_spatial_dims: The number of spatial dimensions of the input.
            output_channels: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``. Defaults to 1.
            rate: Optional kernel dilation rate. Either an integer or a sequence of
                length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
                sequence of n ``(low, high)`` integer pairs that give the padding to
                apply before and after each spatial dimension. or a callable or sequence
                of callables of size ``num_spatial_dims``. Any callables must take a
                single integer argument equal to the effective kernel size and return a
                sequence of two integers representing the padding before and after. See
                ``haiku.pad.*`` for more details and example functions. Defaults to
                ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input.  Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default, ``channels_last``.
            mask: Optional mask of the weights.
            feature_group_count: Optional number of groups in group convolution.
                Default value of 1 corresponds to normal dense convolution. If a higher
                value is used, convolutions are applied separately to that many groups,
                then stacked together. This reduces the number of parameters
                and possibly the compute for a given ``output_channels``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            name: The name of the module.
        """
        super().__init__(name=name)
        if num_spatial_dims <= 0:
            raise ValueError(
                "We only support convolution operations for `num_spatial_dims` "
                f"greater than 0, received num_spatial_dims={num_spatial_dims}."
            )

        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = utils.replicate(
            kernel_shape, num_spatial_dims, "kernel_shape"
        )
        self.with_bias = with_bias
        self.stride = utils.replicate(stride, num_spatial_dims, "strides")
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        self.mask = mask
        self.feature_group_count = feature_group_count
        self.lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
        self.kernel_dilation = utils.replicate(
            rate, num_spatial_dims, "kernel_dilation"
        )
        self.data_format = data_format
        self.channel_index = utils.get_channel_index(data_format)
        self.dimension_numbers = to_dimension_numbers(
            num_spatial_dims, channels_last=(self.channel_index == -1), transpose=False
        )
        self.stochastic_parameters = stochastic_parameters

        if isinstance(padding, str):
            self.padding = padding.upper()
        else:
            self.padding = hk.pad.create(
                padding=padding,
                kernel=self.kernel_shape,
                rate=self.kernel_dilation,
                n=self.num_spatial_dims,
            )

    def __call__(self, inputs: jnp.ndarray, rng_key, stochastic) -> jnp.ndarray:
        """Connects ``ConvND`` layer.
        Args:
            inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
                or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
        Returns:
            An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
                unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
                and rank-N+2 if batched.
        """
        dtype = inputs.dtype

        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        if inputs.shape[self.channel_index] % self.feature_group_count != 0:
            raise ValueError(
                f"Inputs channels {inputs.shape[self.channel_index]} "
                f"should be a multiple of feature_group_count "
                f"{self.feature_group_count}"
            )
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels,
        )

        if self.mask is not None and self.mask.shape != w_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {w_shape}"
            )

        if self.w_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            self.w_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
        if self.b_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            self.b_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)

        w_mu = hk.get_parameter(
            "w_mu", w_shape, dtype, init=self.w_init
        )  ### changed code!

        if self.stochastic_parameters:
            w_logvar = hk.get_parameter(
                "w_logvar", w_shape, dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
            )
            rng_key, sub_key = jax.random.split(rng_key)
            W = gaussian_sample(w_mu, w_logvar, stochastic, sub_key)
            out = lax.conv_general_dilated(
                inputs,
                W,
                window_strides=self.stride,
                padding=self.padding,
                lhs_dilation=self.lhs_dilation,
                rhs_dilation=self.kernel_dilation,
                dimension_numbers=self.dimension_numbers,
                feature_group_count=self.feature_group_count,
            )
        else:
            out = lax.conv_general_dilated(
                inputs,
                w_mu,
                window_strides=self.stride,
                padding=self.padding,
                lhs_dilation=self.lhs_dilation,
                rhs_dilation=self.kernel_dilation,
                dimension_numbers=self.dimension_numbers,
                feature_group_count=self.feature_group_count,
            )

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b_mu = hk.get_parameter("b_mu", bias_shape, inputs.dtype, init=self.b_init)
            if self.stochastic_parameters:
                b_logvar = hk.get_parameter(
                    "b_logvar", shape=bias_shape, dtype=inputs.dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                )
                rng_key, sub_key = jax.random.split(rng_key)
                b = gaussian_sample(b_mu, b_logvar, stochastic, sub_key)
                b = jnp.broadcast_to(b, out.shape)
            else:
                b = jnp.broadcast_to(b_mu, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


def to_dimension_numbers(
    num_spatial_dims: int, channels_last: bool, transpose: bool,
) -> lax.ConvDimensionNumbers:
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if channels_last:
        spatial_dims = tuple(range(1, num_dims - 1))
        image_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        image_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(
        lhs_spec=image_dn, rhs_spec=kernel_dn, out_spec=image_dn
    )


@jit
def partition_params(params):
    '''
    This only works correctly for final_layer_variational=True,
    if other layers are deterministic (i.e., fixed_inner_layers_variational_var=False)
    or dropout is used for inner layers.
    '''
    params_log_var, params_rest = hk.data_structures.partition(predicate_var, params)  # use this if final_layer_variational=False

    def predicate_is_mu_with_log_var(module_name, name, value):
        logvar_name = f"{name.split('_')[0]}_logvar"
        return predicate_mean(module_name, name, value) and \
                module_name in params_log_var and \
                logvar_name in params_log_var[module_name]
    params_mean, params_deterministic = hk.data_structures.partition(predicate_is_mu_with_log_var, params_rest)
    return params_mean, params_log_var, params_deterministic


@jit
def partition_params_final_layer_bnn(params):
    '''
    Use this if final_layer_variational=True and fixed_inner_layers_variational_var=True.
    '''
    params_log_var, params_rest = hk.data_structures.partition(lambda m, n, p: 'logvar' in n and 'linear' in m, params)

    def predicate_is_mu_with_log_var(module_name, name, value):
        logvar_name = f"{name.split('_')[0]}_logvar"
        return predicate_mean(module_name, name, value) and \
                module_name in params_log_var and \
                logvar_name in params_log_var[module_name]
    params_mean, params_deterministic = hk.data_structures.partition(predicate_is_mu_with_log_var, params_rest)
    return params_mean, params_log_var, params_deterministic


@jit
def partition_all_params(params):
    params_mean = hk.data_structures.filter(predicate_mean, params)
    params_batchnorm = hk.data_structures.filter(predicate_batchnorm, params)
    params_log_var = hk.data_structures.filter(predicate_var, params)
    params_rest = params_log_var

    return params_mean, params_batchnorm, params_log_var, params_rest


class BatchNorm(hk.Module):
    """Normalizes inputs to maintain a mean of ~0 and stddev of ~1.

    See: https://arxiv.org/abs/1502.03167.

    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:

    - No scale/offset in which case ``create_*`` should be set to ``False`` and
      ``scale``/``offset`` aren't passed when the module is called.
    - Trainable scale/offset in which case ``create_*`` should be set to
      ``True`` and again ``scale``/``offset`` aren't passed when the module is
      called. In this case this module creates and owns the ``scale``/``offset``
      variables.
    - Externally generated ``scale``/``offset``, such as for conditional
      normalization, in which case ``create_*`` should be set to ``False`` and
      then the values fed in at call time.

    NOTE: ``jax.vmap(hk.transform(BatchNorm))`` will update summary statistics and
    normalize values on a per-batch basis; we currently do *not* support
    normalizing across a batch axis introduced by vmap.
    """

    def __init__(
        self,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        cross_replica_axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        data_format: str = "channels_last",
        name: Optional[str] = None,
        n_condition = None,
        condition_mode = None,
    ):
        """Constructs a BatchNorm module.

        Args:
          create_scale: Whether to include a trainable scaling factor.
          create_offset: Whether to include a trainable offset.
          decay_rate: Decay rate for EMA.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). Can only be set
            if ``create_scale=True``. By default, ``1``.
          offset_init: Optional initializer for bias (aka offset). Can only be set
            if ``create_offset=True``. By default, ``0``.
          axis: Which axes to reduce over. The default (``None``) signifies that all
            but the channel axis should be normalized. Otherwise this is a list of
            axis indices which will have normalization statistics calculated.
          cross_replica_axis: If not ``None``, it should be a string representing
            the axis name over which this module is being run within a ``jax.pmap``.
            Supplying this argument means that batch statistics are calculated
            across all replicas on that axis.
          cross_replica_axis_index_groups: Specifies how devices are grouped.
          data_format: The data format of the input. Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default it is ``channels_last``.
          name: The module name.
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")
        if (cross_replica_axis is None and
            cross_replica_axis_index_groups is not None):
            raise ValueError("`cross_replica_axis` name must be specified"
                             "if `cross_replica_axis_index_groups` are used.")

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
        self.channel_index = utils.get_channel_index(data_format)
        self.mean_ema = hk.ExponentialMovingAverage(decay_rate, name="mean_ema")
        self.var_ema = hk.ExponentialMovingAverage(decay_rate, name="var_ema")

        self.n_condition = n_condition
        self.condition_mode = condition_mode

    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes the normalized version of the input.

        Args:
          inputs: An array, where the data format is ``[..., C]``.
          is_training: Whether this is during training.
          test_local_stats: Whether local stats are used when is_training=False.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
          The array, normalized across all but the last dimension.
        """
        if self.create_scale and scale is not None:
            raise ValueError(
                "Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`.")

        assert len(inputs.shape) >= 2
        if len(inputs.shape) == 2:
            axis = [0]
        else:
            channel_index = self.channel_index
            if channel_index < 0:
                channel_index += inputs.ndim

            if self.axis is not None:
                axis = self.axis
            else:
                axis = [i for i in range(inputs.ndim) if i != channel_index]

        if self.condition_mode == "training_evaluation":
            n_pred = inputs.shape[0] - self.n_condition
            x_condition = inputs[-self.n_condition:]
            x_pred = inputs[:n_pred]

            sum_x_condition = jnp.sum(x_condition, axis, keepdims=True)
            sum_x_condition_sq = jnp.sum(x_condition ** 2, axis, keepdims=True)

            if len(inputs.shape) == 2:
                sum_batch_x_pred = x_pred
                sum_batch_x_pred_sq = x_pred ** 2
            else:
                sum_batch_x_pred = jnp.sum(x_pred, axis[-2:], keepdims=True)
                sum_batch_x_pred_sq = jnp.sum(x_pred ** 2, axis[-2:], keepdims=True)

            sum = sum_x_condition + sum_batch_x_pred
            sum_sq = sum_x_condition_sq + sum_batch_x_pred_sq

            if len(inputs.shape) == 2:
                multiplier = 1
            else:
                multiplier = inputs.shape[1] * inputs.shape[2]

            mean_condition = sum_x_condition / (self.n_condition * multiplier)
            var_condition = sum_x_condition_sq / (self.n_condition * multiplier) - mean_condition ** 2

            mean_batch_pred = sum / ((self.n_condition + 1) * multiplier)
            var_batch_pred = sum_sq / ((self.n_condition + 1) * multiplier) - mean_batch_pred ** 2

            w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
            w_dtype = inputs.dtype

            if self.create_scale:
                scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
            elif scale is None:
                scale = np.ones([], dtype=w_dtype)

            if self.create_offset:
                offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
            elif offset is None:
                offset = np.zeros([], dtype=w_dtype)

            eps = jax.lax.convert_element_type(self.eps, var_batch_pred.dtype)
            inv_condition = scale * jax.lax.rsqrt(var_condition + eps)
            inv_pred = scale * jax.lax.rsqrt(var_batch_pred + eps)

            out_condition = (x_condition - mean_condition) * inv_condition + offset
            out_pred = (x_pred - mean_batch_pred) * inv_pred + offset

            out = jnp.concatenate([out_pred, out_condition], axis=0)

            return out

        else:
            raise ValueError("Condition mode for BatchNorm_mod not specified")
