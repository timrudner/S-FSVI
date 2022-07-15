"""Utilties for partially fill a randomly initialised state/parameters with
the ones resulted from training on the past tasks."""
from typing import Mapping
from typing import NamedTuple
from typing import Sequence
from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
from optax._src.transform import ScaleByAdamState
from optax._src.transform import ScaleState

MODULE = Mapping[str, jnp.ndarray]


def replace_opt_state_trained(
    opt_state_old: Union[Sequence, NamedTuple],
    opt_state_new: Union[Sequence, NamedTuple],
) -> Union[Sequence, NamedTuple]:
    """Replace randomly initialised optimiser state in `opt_state_old` by
    optimiser state resulted from training on the past tasks.

    :param opt_state_old: optax optimiser state resulted from training on a
        sequence of tasks.
    :param opt_state_new: randomly initialised optax optimiser state.
    :return:
        Optimiser state with the same structure as `opt_state_new` but with
        replaced values.
    """
    is_sequence = isinstance(opt_state_old, Sequence)
    opt_state_old = (
        opt_state_old if isinstance(opt_state_old, Sequence) else [opt_state_old]
    )
    opt_state_new = (
        opt_state_new if isinstance(opt_state_new, Sequence) else [opt_state_new]
    )
    assert len(opt_state_old) == len(opt_state_new)
    replaced_states = []
    for i in range(len(opt_state_old)):
        old, new = opt_state_old[i], opt_state_new[i]
        if isinstance(old, ScaleByAdamState):
            assert isinstance(new, ScaleByAdamState)
            new_state = {"count": old.count}
            for field in ["mu", "nu"]:
                new_state[field] = replace_params_trained_heads(
                    params_old=getattr(old, field),
                    params_new=getattr(new, field),
                )
            state = ScaleByAdamState(**new_state)
        elif isinstance(old, ScaleState):
            assert isinstance(new, ScaleState)
            state = old
        else:
            raise NotImplementedError(f"old={old}, new={new}")
        replaced_states.append(state)
    return replaced_states if is_sequence else replaced_states[0]


def replace_params_trained_heads(
    params_old: hk.Params, params_new: hk.Params, final_layer_name: str = None
) -> hk.Params:
    """Replace randomly initialised parameters in `params_new` by trained
    parameters in `params_old`.

    :param params_old: trained parameters.
    :param params_new: newly intialised parameters that have more output heads
        than `params_old`.
    :param final_layer_name: name of the final layer, used to identify the
        last layer.
    :return:
        Parameters with the same structure as `params_new` but with replaced
        values.
    """
    if not final_layer_name:
        final_layer_name = get_omniglot_final_layer_name(params_old)
    is_final_layer = lambda module_name, name, value: module_name == final_layer_name
    final_layers_old, non_final_layers_old = hk.data_structures.partition(
        is_final_layer, params_old
    )
    final_layers_new, _ = hk.data_structures.partition(is_final_layer, params_new)
    new_final_layer_replaced = replace_variables_trained_heads(
        module_old=final_layers_old[final_layer_name],
        module_new=final_layers_new[final_layer_name],
    )
    final_layer_params = hk.data_structures.to_immutable_dict(
        {final_layer_name: new_final_layer_replaced}
    )
    new_params_replaced = hk.data_structures.merge(
        non_final_layers_old, final_layer_params
    )
    return new_params_replaced


def get_omniglot_final_layer_name(params):
    layers = [k for k in params if "linear_final" in k]
    assert len(layers) == 1, layers
    return layers[0]


def replace_variables_trained_heads(
    module_old: MODULE,
    module_new: MODULE,
) -> MODULE:
    """Replace randomly initialised parameters in `module_old` by trained
    parameters in `module_new`.

    Variables in `module_new` are expected to have larger size than the ones
    in `module_old`.

    :param module_old: trained modules.
    :param module_new: newly initialized modules.
    :return:
        Module with the same structure as `module_new` but with replaced
        values.
    """
    module_new_mutable = hk.data_structures.to_mutable_dict(module_new)
    for variable in module_old:
        var_new_transposed = module_new_mutable[variable].T
        var_old_transposed = module_old[variable].T
        indices = jnp.arange(var_old_transposed.shape[0])
        replaced_transposed = jax.ops.index_update(
            var_new_transposed, indices, var_old_transposed
        )
        module_new_mutable[variable] = replaced_transposed.T
    return hk.data_structures.to_immutable_dict(module_new_mutable)
