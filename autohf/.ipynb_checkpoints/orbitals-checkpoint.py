"""
Defining basis sets, atomic orbitals, and molecular orbitals
"""
import jax.numpy as jnp
import basis_set_exchange as bse
from .integrals import atomic_norm

# Functionality for loading default values for basis functions

periodic_table = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4}
# TODO: Finish this


def get_basis_set_symbol(name, symbol):
    e = periodic_table[symbol]
    element = [e]
    basis = bse.get_basis(name, elements=element)['elements']

    exponents = []
    coeffs = []
    L = []
    for f in basis[str(e)]['electron_shells']:
        exponents.append(f['exponents'])
        coeffs.append(f['coefficients'])
        L.append(f['angular_momentum'])

    vals = []

    ao_fn = zip(L, exponents, coeffs)
    for ang_vals, exp_val, c_vals in ao_fn:
        for c, a in zip(c_vals, ang_vals):
            for a_tup in generate_L(a):
                vals.append((a_tup, [float(ex) for ex in exp_val], [float(cx) for cx in c]))

    return vals


def generate_basis_set(name, symbols):
    """
    Generates default basis set parameters
    """
    basis_set = []
    for s in symbols:
        basis_set.append(get_basis_set_symbol(name, s))
    return tuple(basis_set)


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

class AtomicBasisFunction:
    """
    A class representing an atomic basis function. In general, a collection of such
    objects will form a basis set, which can then be used to build molecular orbitals
    """

    def __init__(self, L, R=None, C=None, A=None):
        self.L = L  # Angular momentum tuple
        self.R = R # Location of atomic nucleus
        self.C = C # Contraction coefficients
        self.A = A # Gaussian exponents
        self.params = [R, C, A]

    def __call__(self, *args):
        """
        Calls the molecular orbital
        """
        r = args[0]
        R, C, A = build_param_space(self.params, args[1:])

        l, m, n = self.L
        x0, y0, z0 = R[0], R[1], R[2]
        x, y, z = r[0], r[1], r[2]

        ang = ((x - x0) ** l) * ((y - y0) ** m) * ((z - z0) ** n)
        val = ang * jnp.dot(jnp.array(C), jnp.array([jnp.exp(-alpha * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)) for alpha in A]))
        norm = atomic_norm(self.L, A, C)
        print(norm)
        return norm * val


def get_tuples(length, total):
    """
    Generates all angular momentum tuples corresponding to a value of L
    """
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t

def generate_L(L):
    """
    Generates L-tuples
    """
    return get_tuples(3, L)


class MolecularOrbital:
    """
    A class representing a molecular orbital. Really just a bundle of AtomicBasisFunction classes
    """

    def __init__(self, ao):
        self.coeffs = coeffs
        self.ao = ao

