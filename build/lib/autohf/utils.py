"""
Utility functions
"""
import autograd.numpy as anp

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
    return anp.bincount([anp.ravel_multi_index(index, dims)], None, anp.prod(dims)).reshape(dims)
