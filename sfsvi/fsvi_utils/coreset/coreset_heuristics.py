"""This file contains different methods to select coreset points."""
from typing import Callable, Union, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm

from sfsvi.fsvi_utils.utils_cl import to_float_if_possible
from sfsvi.fsvi_utils.utils_cl import kl_diag as kl_diag_jax
from sfsvi.models.haiku_mod import partition_params
from sfsvi.fsvi_utils.utils_linearization import bnn_linearized_predictive_v2
from sfsvi.fsvi_utils.utils_cl import eps


def add_by_random(x_candidate: np.ndarray, n_add: int) -> np.ndarray:
    """Returns indices of coreset points by random sampling."""
    n_choice = len(x_candidate)
    inds_add = np.random.choice(n_choice, size=n_add, replace=False)
    return inds_add


def add_by_random_per_class(y_candidate: np.ndarray, n_add: int) -> np.ndarray:
    """Returns indices of coreset points by random sampling with equal amount
    of samples from each class."""
    unique_labels = np.unique(y_candidate)
    n_add_per_class = int(np.ceil(n_add / len(unique_labels)))
    inds_add_arrays = []
    for label in unique_labels:
        indices_candidate = np.nonzero(y_candidate == label)[0]
        inds_add_arrays.append(
            np.random.choice(indices_candidate, size=n_add_per_class, replace=False)
        )
    inds_add_all = np.stack(inds_add_arrays).transpose().reshape((-1,))
    inds_add = inds_add_all[:n_add]
    assert len(inds_add) == n_add
    return inds_add


def add_by_entropy(
    x_candidate: np.ndarray,
    n_add: int,
    pred_fn_only_x: Callable,
    mode: str,
    offset: Union[str, float] = 0.0,
    n_mixed: int = 1,
    eval_batch_size: int = 10000,
) -> np.ndarray:
    """Calculate the entropy of predicted class probabilities for each sample,
    then use a certain heuristic to select coreset points from the samples.

    :param x_candidate: a batch of input samples to select coreset points from.
    :param n_add: the number of samples to select from `x_candidate`.
    :param pred_fn_only_x: a function that takes in input and returns MC
        estimated output.
    :param mode: string, the strategy of using entropy to select points.
    :param offset: string "neg_min" (use negative minimum entropy
        as offset) or a float offset to add to entropy before use soft
        selection.
    :param n_mixed: when mode is mixed_highest or mixed_lowest, first use
        hard threshold to shortlist `n_mixed * n_add` points, then use soft selection
        to select `n_add` points.
    :param eval_batch_size: the batch size for evaluation.
    :return:
        indices of coreset points.
    """
    # preds has shape (x_candidate.shape[0], output_dim)
    preds = _make_prediction_by_batch(
        x_candidate=x_candidate,
        pred_fn_only_x=pred_fn_only_x,
        eval_batch_size=eval_batch_size,
    )
    entropy = -np.sum((preds * np.log(preds)), axis=1)
    entropy_quartiles = np.quantile(entropy, (0.25, 0.5, 0.75))
    print("\nEntropy quartiles:", np.around(entropy_quartiles, 4))
    inds_add = select_indices_by_heuristic(
        heuristic=mode, values=entropy, n_add=n_add, offset=offset, n_mixed=n_mixed
    )
    return inds_add


def _make_prediction_by_batch(
    x_candidate: jnp.ndarray,
    pred_fn_only_x: Callable,
    eval_batch_size: int = 10000,
) -> np.ndarray:
    """Returns predicted class probabilities."""
    n_batches = int(np.ceil(x_candidate.shape[0] / eval_batch_size))
    preds_list = []
    for i in range(n_batches):
        preds_list.append(
            pred_fn_only_x(
                x=x_candidate[i * eval_batch_size : (i + 1) * eval_batch_size]
            )
            + eps
        )
    preds = np.concatenate(preds_list, axis=0)
    return preds


def add_by_kl(
    apply_fn,
    stochastic_linearization: bool,
    params,
    state,
    x_candidate: np.ndarray,
    rng_key: jnp.ndarray,
    n_add: int,
    dim_range: Tuple[int, int],
    prior_fn_only_x: Callable,
    heuristic: str = "hard_lowest",
    offset: str = "0.0",
):
    """Calculate the KL divergence term in the ELBO loss for each sample,
    then use a certain heuristic to select coreset points from the samples.
    """
    params_mean, params_log_var, params_deterministic = partition_params(params)
    mean, cov = bnn_linearized_predictive_v2(
        apply_fn,
        params_mean,
        params_log_var,
        params_deterministic,
        state,
        x_candidate,
        rng_key,
        stochastic_linearization,
        full_ntk=False,
        for_loop=True,
    )
    dummy_task_id = 0
    prior_means, prior_covs = prior_fn_only_x(
        context_points={dummy_task_id: x_candidate}
    )
    prior_mean, prior_cov = prior_means[dummy_task_id], prior_covs[dummy_task_id]
    assert (
        mean.shape == cov.shape == prior_cov.shape
    ), f"mean.shape={mean.shape}, cov.shape={cov.shape}, prior_cov.shape={prior_cov.shape}"
    kl_func = jax.vmap(jax.vmap(kl_diag_jax))
    # kl_values has the same shape as prior_mean: (batch_dim, output_dim)
    kl_values = kl_func(
        mean_q=mean,
        mean_p=prior_mean,
        cov_q=cov,
        cov_p=prior_cov,
    )
    kl_of_current_task = kl_values[:, dim_range[0] : dim_range[1]]
    kl_per_sample = jnp.sum(kl_of_current_task, axis=1)
    inds_add = select_indices_by_heuristic(
        heuristic=heuristic, values=kl_per_sample, n_add=n_add, offset=offset
    )
    return inds_add


def add_by_elbo(
    params,
    state,
    rng,
    range_dim: Tuple[int, int],
    x_candidate: np.ndarray,
    y_candidate: np.ndarray,
    loss: Callable,
    n_add: int,
    n_mc_samples: int,
    prior_fn_only_x: Callable,
    heuristic: str,
    offset: str = "0.0",
):
    """Calculate the ELBO for each sample, then use a certain heuristic to
    select coreset points from the samples.
    """
    dummy_task_id = 0
    prior_means, prior_covs = prior_fn_only_x(
        context_points={dummy_task_id: x_candidate}
    )
    prior_mean, prior_cov = prior_means[dummy_task_id], prior_covs[dummy_task_id]
    if len(prior_cov.shape) > 2:
        batch_dim, output_dim = prior_cov.shape[:2]
        prior_cov = jnp.diag(prior_cov.reshape((batch_dim * output_dim, -1))).reshape(
            (batch_dim, output_dim)
        )
    elbos = []

    for i in tqdm(range(x_candidate.shape[0]), desc="evaluating elbos"):
        # this is a hacky way to select the right range of output dimensions
        task_id = 0
        negative_elbo, _ = loss(
            params,
            state,
            {0: prior_mean[i : i + 1]},
            {0: prior_cov[i : i + 1]},
            x_candidate[i : i + 1],
            y_candidate[i : i + 1],
            {0: x_candidate[i : i + 1]},
            rng,
            (range_dim,),
            task_id,
            n_mc_samples,
            False,  # is_training
        )
        elbos.append(-negative_elbo)
    inds_add = select_indices_by_heuristic(
        heuristic=heuristic, values=np.array(elbos), n_add=n_add, offset=offset
    )
    return inds_add


def select_indices_by_heuristic(
    heuristic: str,
    values: np.ndarray,
    n_add: int,
    offset: str = "0.0",
    n_mixed: int = 1,
) -> np.ndarray:
    """Return indices of selected points based on heuristic.

    :param heuristic: string indicating heuristic.
    :param values: values for each sample based on which to select points.
    :param n_add: the number of samples to select from `x_candidate`.
    :param offset: string, the strategy of using entropy to select points.
    :param n_mixed: when mode is mixed_highest or mixed_lowest, first use
        hard threshold to shortlist `n_mixed * n_add` points, then use soft selection
        to select `n_add` points.
    :return:
        indices of selected points.
    """
    values = np.array(values)
    offset = to_float_if_possible(offset)
    if heuristic in ["hard_highest", "highest"]:
        inds_add = np.argsort(values)[-n_add:]
    elif heuristic in ["hard_lowest", "lowest"]:
        inds_add = np.argsort(values)[:n_add]
    elif heuristic == "soft_highest":
        if offset == "neg_min":
            values -= np.min(values)
        else:
            values += offset
        p = values / np.sum(values)
        n_choice = len(values)
        inds_add = np.random.choice(n_choice, size=n_add, replace=False, p=p)
    elif heuristic == "soft_lowest":
        values_reversed = np.max(values) - values
        values_reversed += offset
        p = values_reversed / np.sum(values_reversed)
        n_choice = len(values)
        inds_add = np.random.choice(n_choice, size=n_add, replace=False, p=p)
    # TODO: refactor all mixed heuristics to remove duplicated code
    elif heuristic == "mix_hard_highest_hard_lowest":
        inds_add_highest = select_indices_by_heuristic(
            heuristic="hard_highest", values=values, n_add=n_add // 2, offset=offset
        )
        inds_add_lowest = select_indices_by_heuristic(
            heuristic="hard_lowest",
            values=values,
            n_add=n_add - len(inds_add_highest),
            offset=offset,
        )
        inds_add = np.concatenate([inds_add_highest, inds_add_lowest])
    elif heuristic == "mix_random_soft_lowest":
        inds_add_random = np.random.choice(len(values), size=n_add // 2, replace=False)
        left_index = list(set(list(range(len(values)))) - set(inds_add_random))
        values_left = values[left_index]
        inds_add_soft = select_indices_by_heuristic(
            heuristic="soft_lowest",
            values=values_left,
            n_add=n_add - len(inds_add_random),
            offset=offset,
        )
        inds_add = np.concatenate([inds_add_random, inds_add_soft])
    elif heuristic == "mix_soft_highest_soft_lowest":
        inds_add_highest = select_indices_by_heuristic(
            heuristic="soft_highest", values=values, n_add=n_add // 2, offset=offset
        )
        inds_add_lowest = select_indices_by_heuristic(
            heuristic="soft_lowest",
            values=values,
            n_add=n_add - len(inds_add_highest),
            offset=offset,
        )
        inds_add = np.concatenate([inds_add_highest, inds_add_lowest])
    elif heuristic == "mix_random_soft_highest":
        inds_add_random = np.random.choice(len(values), size=n_add // 2, replace=False)
        left_index = list(set(list(range(len(values)))) - set(inds_add_random))
        values_left = values[left_index]
        inds_add_soft = select_indices_by_heuristic(
            heuristic="soft_highest",
            values=values_left,
            n_add=n_add - len(inds_add_random),
            offset=offset,
        )
        inds_add = np.concatenate([inds_add_random, inds_add_soft])
    elif heuristic == "mixed_highest":
        # Sample `n_add` times from the `n_keep` inputs with highest entropy
        n_keep = n_mixed * n_add
        inds_keep = np.argsort(values)[-n_keep:]
        entropy_keep = values[inds_keep]
        if offset == "neg_min":
            entropy_keep -= np.min(entropy_keep)
        else:
            entropy_keep += offset
        p = entropy_keep / np.sum(entropy_keep)
        inds_add = np.random.choice(inds_keep, size=n_add, replace=False, p=p)
    elif heuristic == "mixed_lowest":
        # Sample `n_add` times from the `n_keep` inputs with lowest entropy
        n_keep = n_mixed * n_add
        inds_keep = np.argsort(values)[:n_keep]
        entropy_keep = values[inds_keep]
        entropy_keep_reversed = np.max(entropy_keep) - entropy_keep
        entropy_keep_reversed += offset
        p = entropy_keep_reversed / np.sum(entropy_keep_reversed)
        inds_add = np.random.choice(inds_keep, size=n_add, replace=False, p=p)
    else:
        raise ValueError(f"Unrecognized selection mode: {heuristic}")
    assert len(inds_add) == n_add
    return inds_add
