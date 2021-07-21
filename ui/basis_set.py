import numpy as np
from basis_data import STO3G

import autograd.numpy as anp
from integrals import atomic_norm, gaussian_norm
from utils import build_param_space


basis_sets = {'sto-3g': STO3G}

s = [(0, 0, 0)]
p = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

def atomic_basis(name, atom):
    
    basis = basis_sets[name][atom]
    params = []
    for i, j in enumerate(basis['orbitals']):
        if j == 'S':
            params.append((s[0], basis['exp'][i], basis['coef'][i]))
        elif j == 'SP':
            for term in j:
                if term == 'S':
                    params.append((s[0], basis['exp'][i], basis['coef'][i]))
                if term == 'P':
                     for l in p:
                        params.append((l, basis['exp'][i], basis['coef'][i+1]))
    return params

def get_basis(name, symbols):
    """
    """
    n_basis = []
    basis_set = []
    for s in symbols:
        basis = atomic_basis(name, s)
        n_basis += [len(basis)]
        basis_set += basis
    return n_basis, tuple(basis_set)


def basis_functions(l, alpha = None, c = None, r = None):
    """ 
    """

    basis_set = [BasisFunction(l[i], r[i], c[i], alpha[i]) for i in range(len(l))]

    return basis_set


class BasisFunction:
    """
    """

    def __init__(self, l, r, c, a):
        self.L = l  
        self.A = anp.array(a)
        self.C = anp.array(c) 
        self.R = anp.array(r) 
        self.params = [anp.array(r), anp.array(c), anp.array(a)]
