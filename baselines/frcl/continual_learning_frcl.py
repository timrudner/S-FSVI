from functools import partial
from typing import Sequence

import gpflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from benchmarking.data_loaders.get_data import make_iterators, prepare_data
from baselines.frcl.utils_frcl import (
    ContinualGPmodel,
    MLPNetworkWithBias,
    evaluate_on_all_tasks,
    predict_at_head_frcl,
    ConvNetworkWithBias)
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class


def train(
    seed,
    task="continual_learning_pmnist",
    batch_size=128,
    hidden_sizes=[100, 100],
    learning_rate=1e-3,
    n_iterations_train=2000,
    n_iterations_discr_search=1000,
    select_method="trace_term",
    use_val_split=True,
    n_permuted_tasks=10,
    logger=None,
    n_coreset_inputs_per_task="not_specified",
    n_omniglot_inducing_chars = 2,
    n_omniglot_tasks = 50,
    randomize_test_split: bool = False,
    randomize_task_sequence: bool = False,
):
    omniglot_test_random_state = seed if randomize_test_split else 0
    omniglot_randomize_task_sequence_seed = seed if randomize_task_sequence else None
    (
        load_task,
        meta_data
    ) = prepare_data(task, use_val_split, n_permuted_tasks=n_permuted_tasks,
                     n_omniglot_coreset_chars=n_omniglot_inducing_chars,
                     omniglot_dtype=np.float64, n_omniglot_tasks=n_omniglot_tasks,
                     omniglot_test_random_state=omniglot_test_random_state,
                     omniglot_randomize_task_sequence_seed=omniglot_randomize_task_sequence_seed,
                     input_dtype=np.float64)
    n_tasks = meta_data["n_tasks"]
    n_inducing_inputs_per_task = meta_data["n_coreset_inputs_per_task_list"]
    input_shape = meta_data["input_shape"]
    output_dim = meta_data["output_dim"]

    if n_coreset_inputs_per_task != "not_specified":
        n_inducing_inputs_per_task = [
            int(n_coreset_inputs_per_task)
        ] * n_tasks

    base_network = compose_base_network(task=task, hidden_sizes=hidden_sizes)

    FRCL = ContinualGPmodel(
        num_features=(hidden_sizes[-1] + 1),
        num_classes=output_dim,
        base_network=base_network,
        likelihood=gpflow.likelihoods.MultiClass(output_dim),
    )

    all_accuracies, all_iterators = [], []
    rng_state = np.random.RandomState(seed)
    for task_id in range(n_tasks):
        print(f"Learning task {task_id+1}")
        if logger:
            logger.info(f"Learning task {task_id+1}")

        task_data = load_task(task_id=task_id)
        task_iterators = make_iterators(task_data, batch_size)
        if use_val_split:
            all_iterators.append(task_iterators.full_valid)
        else:
            all_iterators.append(task_iterators.full_test)

        FRCL.get_weight_space_approx(rng_state)  # weight-space approximation for the current task
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)  # task-specific optimiser

        for _ in tqdm(range(n_iterations_train)):
            x, y = next(task_iterators.batch_train)
            loss = lambda: FRCL.objective_weight_space(x, y, task_data.n_train)[0]
            optimizer.minimize(loss)

        # Randomly select inducing points from the discrete-search set
        x_discr, y_discr = next(task_iterators.full_train)
        permutation_train = np.random.permutation(task_data.n_train)
        n_inputs_to_add = n_inducing_inputs_per_task[task_id]
        inds_inducing = permutation_train[-n_inputs_to_add:]
        x_inducing = tf.gather(x_discr, inds_inducing, axis=0)

        if select_method == "trace_term":
            x_discr_eval, _ = next(task_iterators.full_valid)  # data to evaluate the inducing set on
            loss_current = FRCL.trace_term(x_discr_eval, x_inducing)
            results_discr_search = [(0, loss_current, x_inducing)]
            train_set_id = 0  # candidate training point to replace the inducing point with
            n_accepted_moves = 0

            for iteration in range(n_iterations_discr_search):
                inducing_set_id = iteration % n_inputs_to_add
                inds_inducing_proposed = inds_inducing.copy()

                # Replace inducing point and re-evaluate
                inds_inducing_proposed[inducing_set_id] = permutation_train[train_set_id]
                x_inducing_proposed = tf.gather(x_discr, inds_inducing_proposed, axis=0)
                loss_proposed = FRCL.trace_term(x_discr_eval, x_inducing_proposed)

                if loss_proposed < loss_current:
                    inds_inducing = inds_inducing_proposed  # FIX 3: update `inds_inducing` instead of copying
                    x_inducing = x_inducing_proposed  # FIX 1: update `x_inducing`
                    loss_current = loss_proposed
                    results_discr_search.append((iteration, loss_current, x_inducing))
                    n_accepted_moves += 1

                if train_set_id == task_data.n_train - 1:
                    permutation_train = np.random.permutation(task_data.n_train)
                    train_set_id = 0
                else:
                    train_set_id += 1

            print(f"Discrete search for inducing points: {n_accepted_moves} accepted moves")
            if logger:
                logger.info(f"Discrete search for inducing points: {n_accepted_moves} accepted moves")
            _, _, x_inducing_init = results_discr_search[0]

        elif select_method == "random_choice":
            x_inducing_init = x_inducing
        elif select_method == "random_noise":
            x_inducing = tf.random.uniform((n_inputs_to_add,) + tuple(input_shape[1:]), dtype=tf.float64)
            x_inducing_init = x_inducing
        elif select_method == "random_per_class":
            x_candidate, y_candidate = x_discr.numpy(), y_discr.numpy()
            inds_add = add_by_random_per_class(y_candidate=y_candidate, n_add=n_inputs_to_add)
            print("inducing points")
            print(y_candidate[inds_add])
            if logger:
                logger.info("inducing points")
                logger.info(y_candidate[inds_add])
            x_inducing = tf.convert_to_tensor(x_candidate[inds_add])
            x_inducing_init = x_inducing
        else:
            raise ValueError("Invalid value for select_method")

        FRCL.complete_task_weight_space(x_inducing, x_inducing_init)
        pred_fn = partial(predict_at_head_frcl, model=FRCL)
        accuracies, _ = evaluate_on_all_tasks(all_iterators, pred_fn)
        all_accuracies.append(accuracies)
        split = "val" if use_val_split else "test"
        print(f"Mean accuracy ({split}): {np.mean(accuracies):.4f}")
        print(f"Accuracies ({split}): {accuracies}\n")
        if logger:
            logger.info(f"Mean accuracy ({split}): {np.mean(accuracies):.4f}")
            logger.info(f"Accuracies ({split}): {accuracies}\n")

    return all_accuracies


def compose_base_network(task, hidden_sizes: Sequence[int]):
    if "omniglot" in task:
        recommended_hidden_sizes = [64, 64, 64, 64]
        if hidden_sizes != recommended_hidden_sizes:
            print(f"Not using default hidden sizes in FRCL: "
                  f"default = {recommended_hidden_sizes}, "
                  f"hidden_sizes = {hidden_sizes}")
        base_network = ConvNetworkWithBias(output_sizes=hidden_sizes)
    else:
        base_network = MLPNetworkWithBias(output_sizes=hidden_sizes)
    return base_network
