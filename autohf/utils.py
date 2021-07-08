"""
Utility functions
"""
import jax.numpy as jnp
import numpy as np


def build_param_space(params, args):
    """
    Builds the parameter space
    """
    new_args = []
    counter = 0

    for co, c in enumerate(params):
        if c is None:
            new_args.append(args[counter])
            counter += 1
        else:
            new_args.append(c)
    return tuple(new_args)


def build_arr(dims, index):
    """Builds a matrix"""
    return jnp.bincount(jnp.array([jnp.ravel_multi_index(index, dims)]), None, jnp.prod(jnp.array(dims))).reshape(dims)


def cartesian_prod(*args):
    """Cartesian product of arrays"""
    if len(args) == 1:
        return args[0]
    return cartesian_prod(
        np.transpose([np.tile(args[0], len(args[1])), np.repeat(args[1], len(args[0]))]),
        *args[2:]
    )
