from typing import Callable

import haiku as hk


def map_variable_name(params: hk.Params, fn: Callable) -> hk.Params:
    params = hk.data_structures.to_mutable_dict(params)
    for module in params:
        params[module] = {
            fn(var_name): array for var_name, array in params[module].items()
        }
    return hk.data_structures.to_immutable_dict(params)
