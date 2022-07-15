import pdb
from typing import List, Sequence, Union, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from optax._src.transform import ScaleByAdamState, ScaleState


def replace_opt_state_trained(
    opt_state_old: Union[Sequence, NamedTuple],
    opt_state_new: Union[Sequence, NamedTuple],
) -> Union[Sequence, NamedTuple]:
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
                    params_old=getattr(old, field), params_new=getattr(new, field),
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
    if not final_layer_name:
        final_layer_name = get_omniglot_final_layer_name(params_old)
    predicate = lambda module_name, name, value: module_name == final_layer_name
    final_layers_old, non_final_layers_old = hk.data_structures.partition(
        predicate, params_old
    )
    final_layers_new, _ = hk.data_structures.partition(predicate, params_new)
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


def replace_variables_trained_heads(module_old, module_new):
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
