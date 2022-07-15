"""
The classes here are unedited from the FRCL notebook.
https://github.com/deepmind/deepmind-research/blob/master/functional_regularisation_for_continual_learning/frcl.ipynb
"""
import gpflow
import numpy as np
import sonnet as snt
import tensorflow as tf

from typing import Optional, Sequence, Text, Tuple


eps = 1e-6


# This is copied from `utils_cl.py` (necessary because of difficulties setting
# up an FRCL environment compatible with JAX, TF nightly etc)
def evaluate_on_all_tasks(iterators, pred_fn):
    accuracies, entropies = [], []
    for task_id, iterator in enumerate(iterators):
        x, y = next(iterator)  # whole evaluation dataset for this task
        x, y = np.array(x), np.array(y)
        preds = pred_fn(x=x, task_id=task_id) + eps
        accuracy = np.mean(np.argmax(preds, axis=1) == y)
        entropy = np.mean(-np.sum((preds * np.log(preds)), axis=1))
        accuracies.append(round(accuracy, 4))
        entropies.append(round(entropy, 4))
    return accuracies, entropies


class MLPNetworkWithBias(snt.nets.MLP):
    """Fully connected MLP base network using Sonnet."""

    def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Applies MLP to `inputs` and adds a column of ones to the feature matrix.

        Args:
          inputs: A Tensor of shape `[batch_size, num_input_dimensions]`.
          *args: Arguments to snt.nets.MLP. See Sonnet documentation.
          **kwargs: Named arguments to snt.nets.MLP. See Sonnet documentation.

        Returns:
          outputs: Model output and bias term of shape `[batch_size, output_size+1]`.
        """
        outputs = super(MLPNetworkWithBias, self).__call__(inputs, *args, **kwargs)

        # Add a column of ones to the feature vector to account for the bias
        outputs = tf.concat(
            [outputs, tf.ones((outputs.shape[0], 1), dtype=outputs.dtype)], axis=1
        )

        return outputs


class ConvNetworkWithBias(snt.Module):
  """Builds a module out of a sequence of callables.

  For Sequential Omniglot, we use a good old-fashioned ConvNet with MaxPooling which we define below
  """

  def __init__(self,
               output_sizes: Sequence[int],
               data_format: Text = 'NHWC',
               conv_padding: Text = 'SAME',
               conv_kernel_shape: Sequence[int] = None,
               conv_kernel_stride: Sequence[int] = None,
               maxpool_padding: Text = 'VALID',
               maxpool_kernel_shape: Sequence[int] = None,
               maxpool_kernel_stride: Sequence[int] = None,
               name: Optional[Text] = None):
    """Constructor for ConvNetworkWithBias.

    Args:
      output_sizes: Defines the number of output channels for each ConvLayer.
      data_format: Specifies semantics for each input batch dimension.
      conv_padding: Either `SAME` or `VALID`.
      conv_kernel_shape: Size of the convolutional kernel.
      conv_kernel_stride: Convolution stride.
      maxpool_padding: Either `SAME` or `VALID`.
      maxpool_kernel_shape: Size of the maxpool kernel.
      maxpool_kernel_stride: MaxPool stride.
      name: Optional model name.
    """
    super(ConvNetworkWithBias, self).__init__(name=name)

    if conv_kernel_shape is None:
      conv_kernel_shape = [3, 3]
    if conv_kernel_stride is None:
      conv_kernel_stride = [1, 1]
    if maxpool_padding is None:
      maxpool_padding = 'VALID'
    if maxpool_kernel_shape is None:
      maxpool_kernel_shape = [1, 2, 2, 1]
    if maxpool_kernel_stride is None:
      maxpool_kernel_stride = [1, 2, 2, 1]

    self._output_sizes = output_sizes
    self._num_layers = len(self._output_sizes)
    self._conv_kernel_shapes = [conv_kernel_shape] * self._num_layers
    self._conv_strides = [conv_kernel_stride] * self._num_layers
    self._maxpool_kernel_shapes = [maxpool_kernel_shape] * self._num_layers
    self._maxpool_kernel_strides = [maxpool_kernel_stride] * self._num_layers

    # Instantiate modules
    self._conv_modules = list(
        snt.Conv2D(  # pylint: disable=g-complex-comprehension
            output_channels=self._output_sizes[i],
            kernel_shape=self._conv_kernel_shapes[i],
            stride=self._conv_strides[i],
            padding=conv_padding,
            data_format=data_format,
            name='conv_2d_{}'.format(i))
        for i in range(self._num_layers))

    self._maxpool_modules = list(
        lambda x: tf.nn.max_pool(x,
                                 ksize=self._maxpool_kernel_shapes[i],
                                 strides=self._maxpool_kernel_strides[i],
                                 padding=maxpool_padding,
                                 data_format=data_format,
                                 name='maxpool_2d_{}'.format(i))
        for i in range(self._num_layers))
    self._flatten = snt.Flatten(name='Flatten')

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
      """Applies ConvNet to `inputs` and adds a column of ones to the feature matrix.

      Args:
        inputs: A Tensor of shape `[batch_size, height, width, num_channels]`.

      Returns:
        outputs: Model output and bias term `[batch_size, output_size+1]`.
      """

      # Ensure correct data type
      original_dtype = inputs.dtype
      outputs = tf.cast(inputs, tf.float32)

      for conv_layer, maxpool_layer in zip(
        self._conv_modules, self._maxpool_modules):
          outputs = conv_layer(outputs)
          outputs = maxpool_layer(outputs)
          outputs = tf.nn.relu(outputs)

      outputs = self._flatten(outputs)
      # Add a column of ones to the feature vector to account for the bias
      outputs = tf.concat(
          [outputs, tf.ones((outputs.shape[0], 1), dtype=outputs.dtype)], axis=1)

      return tf.cast(outputs, original_dtype)


class BaseInducingApproximation(object):
    """Holds the parameters of a posterior approximation."""

    def __init__(
        self,
        q_z: tf.Tensor,
        q_mean: tf.Variable,
        q_sqrt: tf.Variable,
        q_z_init: tf.Tensor,
    ):
        """Constructor for BaseInducingApproximation.

        Args:
          q_z: Inducing points. A Tensor of shape `[num_inducing_points, num_input_features]`.
          q_mean: Mean of the variational distribution of shape `[num_inducing_points, num_classes]`.
          q_sqrt: Cholesky (sqrt) matrices of the variational distribution `[]`.
          q_z_init: Initial inducing points. A Tensor of shape `[num_inducing_points, num_input_features]`.
        """
        self.q_z = q_z
        self.q_mean = q_mean
        self.q_sqrt = q_sqrt
        self.q_z_init = q_z_init


class InducingApproximation(BaseInducingApproximation):
    """Creates a variational approximation in function/GP space."""

    def __init__(
        self, num_inducing_points: int, num_classes: int, inducing_init_value: tf.Tensor
    ):
        """Constructor for InducingApproximation.

        Args:
          num_inducing_points: Number of inducing points to use.
          num_classes: Number of classes in classification problem.
          inducing_init_value: Initial value. A Tensor of shape `[num_inducing_points, num_input_features]`.
        """
        q_mean = tf.Variable(
            np.zeros((num_inducing_points, num_classes)).astype(np.float64)
        )
        q_sqrt = tf.Variable(
            np.array(
                [
                    np.eye(num_inducing_points).astype(np.float64)
                    for _ in range(num_classes)
                ]
            ).T
        )

        q_z = tf.constant(inducing_init_value)
        q_z_init = tf.constant(inducing_init_value)
        super(InducingApproximation, self).__init__(q_z, q_mean, q_sqrt, q_z_init)


class WeightSpaceApproximation(object):
    """Creates a variational approximation in weight space."""

    def __init__(self, num_features: int, num_classes: int, rng: np.random.RandomState):
        """Constructor for WeightSpaceApproximation.

        Args:
          num_features: Number of inducing points to use.
          num_classes: Number of classes in classification problem.
          rng: Random number generator.
        """
        scalar = 1.0 / np.sqrt(num_features)
        self.q_w_mean = tf.Variable(
            scalar * rng.randn(num_features, num_classes).astype(np.float64)
        )
        self.q_w_sqrt = tf.Variable(
            np.array(
                [np.eye(num_features).astype(np.float64) for _ in range(num_classes)]
            ).T
        )


class ContinualGPmodel(object):
    """Continual learning GP-based model tha uses variational inducing points."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        base_network: snt.Module,
        likelihood: gpflow.likelihoods,
        noise_variance: float = 1e-3,
    ):
        """Constructor for continual GP model.

        Args:
          num_input_dimensions: Dimensionality of the input space.
          num_features: Size of the feature including the bias term.
          num_classes: Number of classes in classification problem.
          base_network: Provides the feature mappping.
          likelihood: Likelihood for the datatype at hand.
          noise_variance: Added to diagonal of covariance matrix.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.past_inducing_approxs = []
        self.current_inducing_approx = None
        self.current_weight_space_approx = None
        self.base_network = base_network
        self.likelihood = likelihood
        # Constant noise/jitter to be added only to the inducing point covariances
        self.noise_variance = tf.constant(noise_variance, dtype=tf.float64)

    def get_weight_space_approx(self, rng):
        """Initialise a new weight space approximation.

        Args:
          rng: Random number generator.
        """
        self.current_weight_space_approx = WeightSpaceApproximation(
            self.num_features, self.num_classes, rng
        )

    def covariance_self(self, matrix_a: tf.Tensor) -> tf.Tensor:
        """Compute linear kernel plus noise for features with themself.

        Args:
          matrix_a: Matrix of shape `[batch, num_features]`.

        Returns:
          Covariance matrix of shape `[batch, batch]`.
        """
        return tf.matmul(
            matrix_a, matrix_a, transpose_b=True
        ) + self.noise_variance * tf.eye(
            tf.cast(matrix_a.shape[0], tf.int32), dtype=tf.float64
        )

    def covariance_cross(self, matrix_a: tf.Tensor, matrix_b: tf.Tensor) -> tf.Tensor:
        """Compute linear kernel for features with other features.

        No noise in these case because features are assumed distinct.

        Args:
          matrix_a: Tensor of shape `[batch_a, num_features]`.
          matrix_b: Tensor of shape  `[batch_b, num_features]`.

        Returns:
          Covariance matrix of shape `[batch_a, batch_b]`.
        """
        return tf.matmul(matrix_a, matrix_b, transpose_b=True)

    def covariance_diag(self, matrix: tf.Tensor) -> tf.Tensor:
        """Compute diagonal of linear kernel matrix for features with themself.

        Args:
          matrix: Tensor of shape `[batch, num_features]`.

        Returns:
          Diagonal covariance vector of shape `[batch]`.
        """
        # We do not add noise_variance/jitter to this diagonal, but only to the
        # inducing matrix, so inducing variables are noisy function values and
        # through them we approximate the exact noise-free GP model.
        return tf.reduce_sum(matrix * matrix, axis=1)

    def objective_weight_space(
        self, inputs: tf.Tensor, outputs: tf.Tensor, num_task_points: int
    ):
        """Compute the variational objective for the model.

        Args:
          inputs: A Tensor of shape `[batch_size, num_input_dimensions]`.
          outputs: Class labels. Shape (batch, num_classes).
          num_task_points: Total number of data points in task.

        Returns:
          variational objective for the bound.
        """
        batch_size = tf.cast(inputs.shape[0], tf.int32)

        # Add KL divergences for the all previous tasks.
        # Rightmost term of Eq. (4) in the paper.
        kl_historical = 0
        for inducing_approx in self.past_inducing_approxs:
            # z_features shape (num_inducing_points, num_features)
            z_features = self.base_network(inducing_approx.q_z)
            # p_cov shape (num_inducing_points, num_inducing_points)
            p_cov = self.covariance_self(z_features)
            kl_historical += gpflow.kullback_leiblers.gauss_kl(
                inducing_approx.q_mean,
                tf.transpose(inducing_approx.q_sqrt, (2, 0, 1)),
                K=p_cov,
            )

        # For the current task we do inference in the weight space
        data_features = self.base_network(inputs)
        data_mean = tf.matmul(data_features, self.current_weight_space_approx.q_w_mean)

        tr_q_w_sqrt = tf.compat.v1.matrix_band_part(
            tf.transpose(self.current_weight_space_approx.q_w_sqrt, (2, 0, 1)), -1, 0
        )
        expand_data_features = tf.tile(
            tf.expand_dims(data_features, 0), [self.num_classes, 1, 1]
        )
        feat_w_sqrt = tf.matmul(expand_data_features, tr_q_w_sqrt)
        data_var = tf.transpose(tf.reduce_sum(tf.square(feat_w_sqrt), axis=2))

        # Middle term of Eq. (4) in the paper.
        kl_current = gpflow.kullback_leiblers.gauss_kl(
            self.current_weight_space_approx.q_w_mean,
            tf.transpose(self.current_weight_space_approx.q_w_sqrt, (2, 0, 1)),
            K=None,
        )

        # Left term of Eq. (4) in the paper.
        exp_likes = self.likelihood.variational_expectations(
            data_mean, data_var, outputs
        )
        # Scale this correctly by the total number of datapoints in the task.
        ratio = tf.cast(num_task_points, tf.float64) / tf.cast(batch_size, tf.float64)
        reconstruction_cost = tf.reduce_sum(exp_likes) * ratio

        # Full objective. Eq. (4) in the paper.
        objective = kl_historical + kl_current - reconstruction_cost
        return objective, reconstruction_cost, kl_current

    def trace_term(self, inputs: tf.Tensor, q_z: tf.Tensor) -> tf.Tensor:
        """Compute the trace term. Useful for search over inducing points.

        Args:
          inputs: A Tensor of shape `[batch_size, num_input_dimensions]`.
          q_z: Inducing points of shape `[num_inducing_points, num_input_dimensions]`.

        Returns:
          variational objective for the bound.
        """
        # Shape [num_inducing_points, num_features]
        inducing_features = self.base_network(q_z)
        # Shape [batch_size, num_features]
        data_features = self.base_network(inputs)

        # Compute covariance
        kmn = self.covariance_cross(inducing_features, data_features)
        kmm = self.covariance_self(inducing_features)
        knn = self.covariance_diag(data_features)

        # Implements Equation (6) in the paper and speeds up computations by
        # applying the matrix inversion formula if needed.
        if q_z.shape[0] <= self.num_features:
            lm = tf.compat.v1.cholesky(kmm)
            v = tf.compat.v1.matrix_triangular_solve(lm, kmn, lower=True)
            fvar = knn - tf.reduce_sum(tf.square(v), 0)
        else:
            phimphim = tf.matmul(inducing_features, inducing_features, transpose_a=True)
            phimphim_noise = phimphim + self.noise_variance * tf.eye(
                tf.cast(self.num_features, tf.int32), dtype=tf.float64
            )

            sqrt_phimphin_noise = tf.compat.v1.cholesky(phimphim_noise)

            inv_sqrt_phimphin_noise_phim = tf.compat.v1.matrix_triangular_solve(
                sqrt_phimphin_noise, tf.transpose(inducing_features), lower=True
            )
            v = tf.matmul(inv_sqrt_phimphin_noise_phim, kmn)

            fvar = knn - (1.0 / self.noise_variance) * (
                tf.reduce_sum(tf.square(kmn), 0) - tf.reduce_sum(tf.square(v), 0)
            )

        return tf.reduce_sum(fvar)

    def function_space_prediction(
        self,
        inputs: tf.Tensor,
        q_z: tf.Tensor,
        q_mean: tf.Variable,
        q_sqrt: tf.Variable,
        outputs: tf.Tensor = None,
    ):
        """Compute predictions from the model in function/GP space.

        Args:
          inputs: Tensor of shape `[batch, num_input_dimensions]`.
          q_z: Inducing points of shape `[num_inducing_points, input_dimensions]`.
          q_mean: Mean of the variational distribution of shape `[num_inducing_points, num_classes]`.
          q_sqrt: Cholesky (sqrt) matrices of the variational distribution `[]`.
          outputs: Class labels. Optional tensor of Shape `[batch, num_clsses]`.

        Returns:
          When outputs=None, it returns predictive mean and variance from model
          otherwise it returns the log predictive density for the outputs.
        """
        inducing_features = self.base_network(q_z)
        data_features = self.base_network(inputs)

        # Compute covariance
        kmn = self.covariance_cross(inducing_features, data_features)
        kmm = self.covariance_self(inducing_features)
        knn = self.covariance_diag(data_features)

        func_mean, func_var = gpflow.conditionals.base_conditional(
            kmn,
            kmm,
            knn,
            f=q_mean,
            q_sqrt=tf.transpose(q_sqrt, (2, 0, 1)),
            full_cov=False,
            white=False,
        )

        if outputs is None:
            return self.likelihood.predict_mean_and_var(func_mean, func_var)
        else:
            return self.likelihood.predict_density(func_mean, func_var, outputs)

    def complete_task_weight_space(self, z: tf.Tensor, z_init: tf.Tensor):
        """Completes training of current task in weight space.

        Args:
          z: Final inducing points for current task of shape `[num_inducing_points, num_features]`.
          z_init: Initial inducing points for current task of shape  `[num_inducing_points, num_features]`.
        """
        current_inducing_features = self.base_network(z)
        current_q_mean = tf.matmul(
            current_inducing_features, self.current_weight_space_approx.q_w_mean
        )
        noise_matrix = self.noise_variance * tf.eye(
            tf.cast(tf.shape(z)[0], tf.int32), dtype=tf.float64
        )
        noise_matrix = tf.tile(
            tf.expand_dims(noise_matrix, 0), [self.num_classes, 1, 1]
        )
        tr_q_w_sqrt = tf.linalg.band_part(
            tf.transpose(self.current_weight_space_approx.q_w_sqrt, (2, 0, 1)), -1, 0
        )
        expand_current_inducing_features = tf.tile(
            tf.expand_dims(current_inducing_features, 0), [self.num_classes, 1, 1]
        )
        feat_w_sqrt = tf.matmul(expand_current_inducing_features, tr_q_w_sqrt)
        q_cov = tf.matmul(feat_w_sqrt, feat_w_sqrt, transpose_b=True)
        q_cov = q_cov + noise_matrix
        current_q_sqrt = tf.transpose(tf.compat.v1.cholesky(q_cov), (1, 2, 0))

        self.past_inducing_approxs.append(
            BaseInducingApproximation(
                z, tf.identity(current_q_mean), tf.identity(current_q_sqrt), z_init
            )
        )


def predict_at_head_frcl(x, task_id, model: ContinualGPmodel):
    preds, _ = model.function_space_prediction(
        x,
        model.past_inducing_approxs[task_id].q_z,
        model.past_inducing_approxs[task_id].q_mean,
        model.past_inducing_approxs[task_id].q_sqrt,
    )
    return preds
