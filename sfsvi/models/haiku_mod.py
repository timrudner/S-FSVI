"""Stochastic versions of haiku convolutional and fully connected layers."""
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src.utils import get_channel_index
from haiku._src.utils import replicate
from jax import jit
from jax import lax
from jax import random


class DenseStochastic(hk.Module):
    """Dense layer that is by default stochastic."""

    def __init__(
        self,
        output_size: int,
        init_logvar_minval: float,
        init_logvar_maxval: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        stochastic_parameters: bool = True,
    ):
        """
        :param output_size: the number of hidden units.
        :param init_logvar_minval: lower bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param init_logvar_maxval: upper bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param with_bias: if True, include bias parameters.
        :param w_init: initialiser for weight parameters.
        :param b_init: initialiser for bias parameters.
        :param name: name of the module.
        :param stochastic_parameters: if True, instantiate a stochastic
            layer instead of deterministic layer.
        """
        super(DenseStochastic, self).__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = init_logvar_minval
        self.uniform_init_maxval = init_logvar_maxval
        self.stochastic_parameters = stochastic_parameters

    def __call__(
        self, inputs: jnp.ndarray, rng_key: jnp.ndarray, stochastic: bool
    ) -> jnp.ndarray:
        """Forward pass of the fully connected layer.

        :param inputs: input array.
        :param rng_key: JAX random key.
        :param stochastic: if True, use sampled parameters, otherwise, use mean
            parameters.
        :return:
            output of the fully connected layer.
        """
        j, k = inputs.shape[-1], self.output_size
        self.input_size = inputs.shape[-1]
        dtype = inputs.dtype

        if self.w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            self.w_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
        w_mu = hk.get_parameter("w_mu", shape=[j, k], dtype=dtype, init=self.w_init)

        if self.with_bias:
            if self.b_init is None:
                stddev = 1.0 / np.sqrt(self.input_size)
                self.b_init = hk.initializers.RandomUniform(
                    minval=-stddev, maxval=stddev
                )
            b_mu = hk.get_parameter("b_mu", shape=[k], dtype=dtype, init=self.b_init)
        if self.stochastic_parameters:
            w_logvar = hk.get_parameter(
                "w_logvar",
                shape=[j, k],
                dtype=dtype,
                init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval),
            )
            if self.with_bias:
                b_logvar = hk.get_parameter(
                    "b_logvar",
                    shape=[k],
                    dtype=dtype,
                    init=uniform_mod(
                        self.uniform_init_minval, self.uniform_init_maxval
                    ),
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


class Conv2DStochastic(hk.Module):
    """2D convolution layer that is by default stochastic."""

    def __init__(
        self,
        output_channels: int,
        init_logvar_minval: float,
        init_logvar_maxval: float,
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
        stochastic_parameters: bool = True,
    ):
        """Initializes the module.

        :param output_channels: Number of output channels.
        :param init_logvar_minval: lower bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param init_logvar_maxval: upper bound of the range from which to
            uniformly sample the initial value of log of variational variance
            parameters.
        :param kernel_shape: The shape of the kernel. Either an integer or a sequence of
            length ``num_spatial_dims``.
        :param num_spatial_dims: The number of spatial dimensions of the input.

        :param stride: Optional stride for the kernel. Either an integer or a sequence of
            length ``num_spatial_dims``. Defaults to 1.
        :param rate: Optional kernel dilation rate. Either an integer or a sequence of
            length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
            ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
        :param padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
            sequence of n ``(low, high)`` integer pairs that give the padding to
            apply before and after each spatial dimension. or a callable or sequence
            of callables of size ``num_spatial_dims``. Any callables must take a
            single integer argument equal to the effective kernel size and return a
            sequence of two integers representing the padding before and after. See
            ``haiku.pad.*`` for more details and example functions. Defaults to
            ``SAME``. See:
            https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
        :param with_bias: Whether to add a bias. By default, true.
        :param w_init: Optional weight initialization. By default, truncated normal.
        :param b_init: Optional bias initialization. By default, zeros.
        :param data_format: The data format of the input.  Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default, ``channels_last``.
        :param mask: Optional mask of the weights.
        :param feature_group_count: Optional number of groups in group convolution.
            Default value of 1 corresponds to normal dense convolution. If a higher
            value is used, convolutions are applied separately to that many groups,
            then stacked together. This reduces the number of parameters
            and possibly the compute for a given ``output_channels``. See:
            https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
        :param name: The name of the module.
        :param stochastic_parameters: if True, instantiate a stochastic
            layer instead of deterministic layer.
        """
        super().__init__(name=name)
        if num_spatial_dims <= 0:
            raise ValueError(
                "We only support convolution operations for `num_spatial_dims` "
                f"greater than 0, received num_spatial_dims={num_spatial_dims}."
            )
        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = replicate(kernel_shape, num_spatial_dims, "kernel_shape")
        self.with_bias = with_bias
        self.stride = replicate(stride, num_spatial_dims, "strides")
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = init_logvar_minval
        self.uniform_init_maxval = init_logvar_maxval
        self.mask = mask
        self.feature_group_count = feature_group_count
        self.lhs_dilation = replicate(1, num_spatial_dims, "lhs_dilation")
        self.kernel_dilation = replicate(rate, num_spatial_dims, "kernel_dilation")
        self.data_format = data_format
        self.channel_index = get_channel_index(data_format)
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

        :param inputs:
            An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
            or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.

        :return
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

        w_mu = hk.get_parameter("w_mu", w_shape, dtype, init=self.w_init)

        if self.stochastic_parameters:
            w_logvar = hk.get_parameter(
                "w_logvar",
                w_shape,
                dtype=dtype,
                init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval),
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
                    "b_logvar",
                    shape=bias_shape,
                    dtype=inputs.dtype,
                    init=uniform_mod(
                        self.uniform_init_minval, self.uniform_init_maxval
                    ),
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


def uniform_mod(min_val: float, max_val: float) -> Callable:
    """Returns a weight initializer for uniform sampling from the given range.

    :param min_val: lower bound of the range.
    :param max_val: upper bound of the range.
    :return:
        weight initializer.
    """

    def _uniform_mod(shape, dtype):
        rng_key, _ = random.split(random.PRNGKey(0))
        return jax.random.uniform(
            rng_key, shape=shape, dtype=dtype, minval=min_val, maxval=max_val
        )

    return _uniform_mod


def gaussian_sample(
    mu: jnp.ndarray, rho: jnp.ndarray, stochastic: bool, rng_key: jnp.ndarray
) -> jnp.ndarray:
    """Return a Gaussian sample.

    :param mu: mean of Gaussian.
    :param rho: log variance of Gaussian.
    :param stochastic: if True, then treat the variance to be 0.
    :param rng_key: JAX random key.
    :return:
        Gaussian sample.
    """
    if stochastic:
        jnp_eps = random.normal(rng_key, mu.shape)
        z = mu + jnp.exp((0.5 * rho).astype(jnp.float32)) * jnp_eps
    else:
        z = mu
    return z


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool,
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
def partition_params(params: hk.Params) -> Tuple[hk.Params, ...]:
    """
    This only works correctly for final_layer_variational=True,
    if other layers are deterministic
    or dropout is used for inner layers.
    """
    params_log_var, params_rest = hk.data_structures.partition(
        predicate_var, params
    )  # use this if final_layer_variational=False

    def predicate_is_mu_with_log_var(module_name, name, value):
        logvar_name = f"{name.split('_')[0]}_logvar"
        return (
            predicate_mean(module_name, name, value)
            and module_name in params_log_var
            and logvar_name in params_log_var[module_name]
        )

    params_mean, params_deterministic = hk.data_structures.partition(
        predicate_is_mu_with_log_var, params_rest
    )
    return params_mean, params_log_var, params_deterministic


def predicate_mean(module_name: str, name: str, value: Any) -> bool:
    return name == "w_mu" or name == "b_mu"


def predicate_var(module_name: str, name: str, value: Any) -> bool:
    return name == "w_logvar" or name == "b_logvar"


class KeyHelper:

    def __init__(self, key):
        self._key = key

    def next_key(self):
        self._key, sub_key = jax.random.split(self._key)
        return sub_key
