"""Utilities for processing and loading data for Sequential Omniglot."""
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


def get_omniglot(
    fix_shuffle: bool,
    use_val_split: bool,
    n_omniglot_tasks: int,
    omniglot_dtype,
    omniglot_randomize_task_sequence_seed: int,
    n_omniglot_coreset_chars: int,
    omniglot_test_random_state: int,
    test_set_split: float = 0.2,
    # TODO: think about where to cache the datases
    data_dir: str = "~/.keras/datasets",
    use_data_augmentation: bool = True,
    augmentation_factor: int = 20,
    data_augmentation_seed=None,
    **kwargs,
):
    """Returns data and meta data for Sequential Omniglot."""
    x, y, alphabet_ids = _load_data(data_dir)
    x = _preprocess_x(x, omniglot_dtype)
    # Find and count the unique labels in each alphabet
    if omniglot_randomize_task_sequence_seed is not None:
        alphabet_ids = _shuffle_alphabet_ids(
            alphabet_ids, omniglot_randomize_task_sequence_seed
        )
    x, y, alphabet_ids = _discard_unused_tasks(x, y, alphabet_ids, n_omniglot_tasks)
    y, alphabet_sizes, labels_in_alphabets = _recalculate_global_labels(
        y, alphabet_ids, n_omniglot_tasks
    )

    data = _split_data_into_tasks(
        x=x,
        y=y,
        use_val_split=use_val_split,
        test_set_split=test_set_split,
        alphabet_ids=alphabet_ids,
        n_omniglot_tasks=n_omniglot_tasks,
        use_data_augmentation=use_data_augmentation,
        omniglot_test_random_state=omniglot_test_random_state,
        augmentation_factor=augmentation_factor,
        fix_shuffle=fix_shuffle,
        data_augmentation_seed=data_augmentation_seed,
    )

    # Use multiple heads for Omniglot
    range_dims_per_task, max_entropies = [], []
    for labels in labels_in_alphabets:
        range_dims_per_task.append((min(labels), max(labels) + 1))
        max_entropies.append(-np.log(1 / len(range_dims_per_task[-1])))
    range_dims_per_task = tuple(range_dims_per_task)
    max_entropies = tuple(max_entropies)

    meta_data = {
        "n_tasks": n_omniglot_tasks,
        "n_train": _calculate_number_of_training_points(
            y=y,
            test_set_split=test_set_split,
            use_val_split=use_val_split,
            augmentation_factor=augmentation_factor,
            use_data_augmentation=use_data_augmentation,
        ),
        "input_shape": [1] + list(x.shape[1:]),
        "output_dim": sum(alphabet_sizes),
        "n_coreset_inputs_per_task_list": _calculate_coreset_sizes(
            alphabet_sizes, n_omniglot_coreset_chars
        ),
        "range_dims_per_task": range_dims_per_task,
        "max_entropies": max_entropies,
    }
    return data, meta_data


def _load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw Omniglot data."""
    kwargs_load = {
        "name": "omniglot",
        "data_dir": data_dir,
        "batch_size": -1,
    }
    ds_train = tfds.as_numpy(tfds.load(split="train", **kwargs_load))
    ds_test = tfds.as_numpy(tfds.load(split="test", **kwargs_load))
    x = np.concatenate((ds_train["image"], ds_test["image"]))
    y = np.concatenate((ds_train["label"], ds_test["label"]))
    # alphabet_ids is a 1-D array containing the alphabet_id of each sample
    alphabet_ids = np.concatenate((ds_train["alphabet"], ds_test["alphabet"]))
    return x, y, alphabet_ids


def _preprocess_x(x: np.ndarray, omniglot_dtype) -> np.ndarray:
    """Apply data processing to input images."""
    x = tf.image.convert_image_dtype(x, np.float32)  # Use float32 even for FRCL
    x = tf.image.rgb_to_grayscale(x)
    x = 1.0 - x  # Black on white -> white on black
    x = tf.image.resize(x, (28, 28))
    x = x.numpy().astype(omniglot_dtype)
    return x


def _shuffle_alphabet_ids(alphabet_ids: np.ndarray, seed: int) -> np.ndarray:
    rng_state = np.random.RandomState(seed)
    old_aids_to_new_aids = rng_state.permutation(len(np.unique(alphabet_ids)))
    alphabet_ids = np.array([old_aids_to_new_aids[_aid] for _aid in alphabet_ids])
    return alphabet_ids


def _recalculate_global_labels(
    y: np.ndarray,
    alphabet_ids: np.ndarray,
    n_omniglot_tasks: int,
) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    """"""
    # `label_map` maps from the original label to new label
    # `labels_in_alphabets` is new class labels for each sample
    class_counter = 0
    labels_in_alphabets, alphabet_sizes, label_map = [], [], {}
    for alphabet_id in range(n_omniglot_tasks):
        inds_alphabet_id = np.flatnonzero(alphabet_ids == alphabet_id)
        labels_alphabet_id = np.unique(y[inds_alphabet_id])
        alphabet_sizes.append(len(labels_alphabet_id))
        for label in labels_alphabet_id:
            label_map[label] = class_counter
            class_counter += 1
        labels_alphabet_id = [label_map[label] for label in labels_alphabet_id]
        labels_in_alphabets.append(np.array(labels_alphabet_id))
    y = np.array([label_map[y_i] for y_i in y])
    return y, alphabet_sizes, labels_in_alphabets


def _discard_unused_tasks(
    x: np.ndarray,
    y: np.ndarray,
    alphabet_ids: np.ndarray,
    n_omniglot_tasks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard tasks that are not selected."""
    # If n_tasks < 50, discard the data from the unused alphabets
    inds_keep = np.flatnonzero(alphabet_ids < n_omniglot_tasks)
    x = x[inds_keep]
    y = y[inds_keep]
    alphabet_ids = alphabet_ids[inds_keep]
    return x, y, alphabet_ids


def _calculate_coreset_sizes(
    alphabet_sizes: np.ndarray,
    n_omniglot_coreset_chars: int,
):
    """Compute number of coreset points for each task.

    Use a coreset size proportional to the alphabet size (#characters)
    """
    n_coreset_inputs_per_task_list = []
    for alphabet_size in alphabet_sizes:
        n_coreset_inputs_per_task_list.append(n_omniglot_coreset_chars * alphabet_size)
    n_coreset_inputs_per_task_list = tuple(n_coreset_inputs_per_task_list)
    return n_coreset_inputs_per_task_list


def _split_data_into_tasks(
    x: np.ndarray,
    y: np.ndarray,
    use_val_split: bool,
    test_set_split: float,
    alphabet_ids: np.ndarray,
    n_omniglot_tasks: int,
    use_data_augmentation: bool,
    omniglot_test_random_state: int,
    augmentation_factor: int,
    fix_shuffle: bool,
    data_augmentation_seed: int,
):
    """Generate data for each task in the Sequential Omniglot task sequence."""
    data = []
    for task_id in range(n_omniglot_tasks):
        inds_task = np.flatnonzero(alphabet_ids == task_id)
        x_t = x[inds_task]
        y_t = y[inds_task]
        # Use a fixed train-test split but a controllable train-val split
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x_t,
            y_t,
            test_size=test_set_split,
            random_state=omniglot_test_random_state,
            shuffle=True,
            stratify=y_t,
        )
        if use_val_split:
            (
                x_train,
                x_valid,
                y_train,
                y_valid,
            ) = sklearn.model_selection.train_test_split(
                x_train,
                y_train,
                test_size=len(x_test),
                random_state=(0 if fix_shuffle else None),
                shuffle=True,
                stratify=y_train,
            )
        if use_data_augmentation:
            # Apply the data augmentation proposed in the 'Progress & compress' paper
            x_train_aug = [x_train]
            for _ in range(augmentation_factor - 1):
                x_train_aug.append(
                    apply_random_shift_and_rotation(
                        x_train, seed=data_augmentation_seed
                    )
                )
            x_train = np.concatenate(x_train_aug)
            y_train = np.tile(y_train, (20,))

        if use_val_split:
            data_t = {
                "train": [x_train, y_train],
                "valid": [x_valid, y_valid],
                "test": [x_test, y_test],
            }
        else:
            data_t = {
                "train": [x_train, y_train],
                "test": [x_test, y_test],
            }
        data.append(data_t)
    return data


def _calculate_number_of_training_points(
    y: np.ndarray,
    test_set_split: float,
    use_val_split: bool,
    augmentation_factor: int,
    use_data_augmentation: bool,
):
    n_train = (
        int((1 - 2 * test_set_split) * len(y))
        if use_val_split
        else int((1 - test_set_split) * len(y))
    )
    n_train = (augmentation_factor * n_train) if use_data_augmentation else n_train
    return n_train


def apply_random_shift_and_rotation(
    x: np.array,
    degree_range: Sequence[float] = None,
    shift_range: Sequence[float] = None,
    seed=None,
) -> np.array:
    """Augment images by random shifts and rotations.

    :param x: tensor of shape (n_images, n_rows, n_columns, n_channels).
    :param degree_range: min/max degrees by which each image is rotated.
    :param shift_range: min/max distance by which each image is shifted.
    :return:
        x: augmented image with the same shape as the input.
    """
    if degree_range is None:
        degree_range = (-30.0, 30.0)
    if shift_range is None:
        shift_range = (-5.0, 5.0)
    n = x.shape[0]
    if seed:
        rng_state = np.random.RandomState(seed)
        angles = np.radians(rng_state.uniform(*degree_range, size=n))
        distances = rng_state.uniform(*shift_range, size=2 * n).reshape(n, 2)
    else:
        angles = np.radians(np.random.uniform(*degree_range, size=n))
        distances = np.random.uniform(*shift_range, size=2 * n).reshape(n, 2)
    shifts = np.tile((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), (n, 1))
    shifts = shifts.astype(np.float32)
    shifts[:, 2] = -distances[:, 0]
    shifts[:, 5] = -distances[:, 1]
    x = tfa.image.rotate(x, angles)
    x = tfa.image.transform(x, shifts)
    return x.numpy()
