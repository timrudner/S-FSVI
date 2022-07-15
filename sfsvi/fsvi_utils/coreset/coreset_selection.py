from functools import partial
from typing import Callable

import numpy as np
import haiku as hk

from sfsvi.models.networks import Model
from sfsvi.general_utils.jax_utils import KeyHelper
from sfsvi.fsvi_utils.utils_cl import TUPLE_OF_TWO_TUPLES
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random, add_by_random_per_class, add_by_entropy, add_by_kl, \
    add_by_elbo
from sfsvi.fsvi_utils.prior import CLPrior
from sfsvi.fsvi_utils.utils_cl import predict_at_head_fsvi


def get_coreset_indices(
    hparams,
    x_candidate: np.ndarray,
    y_candidate: np.ndarray,
    n_add: int,
    model: Model,
    params: hk.Params,
    state: hk.State,
    kh: KeyHelper,
    range_dims_per_task: TUPLE_OF_TWO_TUPLES,
    task_id: int,
    prior: CLPrior,
    apply_fn: Callable,
    params_prior: hk.Params,
    loss: Callable,
    stochastic_linearization: bool,
):
    if hparams.coreset == "random":
        print("Adding context points to the coreset randomly")
        inds_add = add_by_random(x_candidate, n_add)
    elif hparams.coreset == "random_per_class":
        inds_add = add_by_random_per_class(y_candidate=y_candidate, n_add=n_add)
        print(
            f"Adding context points to the coreset randomly, {len(inds_add)} samples in total, "
            f"with each class getting the same number of samples"
        )
    elif hparams.coreset == "entropy":
        print("Adding context points to the coreset based on predictive entropy")
        options = {
            "mode": hparams.coreset_entropy_mode,
            "offset": float(hparams.coreset_entropy_offset),
            "n_mixed": hparams.coreset_entropy_n_mixed,
        }
        print(f"Options: {options}")
        pred_fn = make_pred_fn(
            model=model,
            params=params,
            state=state,
            rng_key=kh.next_key(),
            n_samples_eval=hparams.n_samples_eval,
            range_dims_per_task=range_dims_per_task,
        )
        pred_fn_only_x = partial(pred_fn, task_id=task_id)
        inds_add = add_by_entropy(
            x_candidate, n_add, pred_fn_only_x, **options
        )
    elif hparams.coreset in ["kl", "elbo"]:
        if task_id == 0:
            # The prior function we get when calling `training.kl_input_functions()`
            # with `prior_type="fixed"` causes an error in the KL function, so just
            # use randomly selected points for the first task
            print("Adding context points to the coreset randomly")
            inds_add = add_by_random(x_candidate, n_add)
        else:
            print(f"Adding context points to the coreset based on {hparams.coreset}")
            prior_fn = prior.make_prior_fn(
                apply_fn=apply_fn,
                state=state,
                params=params_prior,
                rng_key=kh.next_key(),
                task_id=task_id,
                jit_prior=False,  # Otherwise we get a JAX error
                for_loop=True,  # this optioin is fast for forward pass
            )
            # kl_fn is a function of prior_mean, prior_cov, inputs, context_points
            assert not hparams.full_ntk
            if hparams.coreset == "kl":
                inds_add = add_by_kl(
                    apply_fn=apply_fn,
                    stochastic_linearization=stochastic_linearization,
                    params=params,
                    state=state,
                    x_candidate=x_candidate,
                    rng_key=kh.next_key(),
                    n_add=n_add,
                    prior_fn_only_x=prior_fn,
                    heuristic=hparams.coreset_kl_heuristic,
                    offset=hparams.coreset_kl_offset,
                    dim_range=range_dims_per_task[task_id],
                )
            elif hparams.coreset == "elbo":
                inds_add = add_by_elbo(
                    params=params,
                    state=state,
                    rng=kh.next_key(),
                    range_dim=range_dims_per_task[task_id],
                    x_candidate=x_candidate,
                    y_candidate=y_candidate,
                    loss=loss,
                    prior_fn_only_x=prior_fn,
                    heuristic=hparams.coreset_elbo_heuristic,
                    offset=hparams.coreset_elbo_offset,
                    n_add=n_add,
                    n_mc_samples=hparams.coreset_elbo_n_samples,
                )
    else:
        raise ValueError(f"Invalid value for hparams.coreset: {hparams.coreset}")
    return inds_add


def make_pred_fn(
    model,
    params,
    state,
    rng_key,
    n_samples_eval: int,
    range_dims_per_task: TUPLE_OF_TWO_TUPLES,
) -> Callable:
    """
    Return a function that output MC estimate of model output

    @param n_samples_eval: number of MC samples to estimate the mean output, which is used to evaluate
        accuracies
    @return
        a function that takes in two keyword arguments
            x: input data
            task_id: the index of task
        and returns the MC estimate of model output
    """
    # `pred_fn` is callable with `x` arg
    pred_fn = partial(
        model.predict_f_multisample,
        params=params,
        params_feature=params,
        state=state,
        rng_key=rng_key,
        n_samples=n_samples_eval,
        is_training=False,
    )

    # `pred_fn` is callable with `x` and `task_id` args
    _pred_fn = partial(
        predict_at_head_fsvi, pred_fn=pred_fn, range_dims_per_task=range_dims_per_task,
    )

    return _pred_fn