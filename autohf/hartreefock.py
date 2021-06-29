"""
Methods for performing Hartree-Fock
"""
from .integrals import *
import jax
import jax.numpy as jnp
from tqdm import tqdm


def overlap_matrix(atomic_orbitals):
    """
    Generates the overlap matrix
    """
    def overlap(*args):
        S = jnp.zeros((len(atomic_orbitals), len(atomic_orbitals)))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i < j:
                    overlap_integral = generate_overlap(a, b)(args[i], args[j])
                    S = jax.ops.index_update(S, jax.ops.index[(i, j), (j, i)], overlap_integral)
                if i == j:
                    S = jax.ops.index_update(S, jax.ops.index[i, i], 1.0)
        return S
    return overlap


def density_matrix(C):
    """
    Computes the density matrix
    """
    C = jnp.array(C)
    """
    P = jnp.zeros((len(C), len(C)))

    for i in range(len(C)):
        for j in range(len(C)):
            if i <= j:
                entry = jnp.dot(C[i], C[j])
                P[i][j], P[j][i] = entry, entry
    """
    # TODO: Figure out what is going on here...
    return jnp.dot(C[:,:1],jnp.conjugate(C[:,:1]).T)


def electron_repulsion_tensor(atomic_orbitals):
    # TODO: Optimize this more!
    """Computes a tensor of electron repulsion integrals"""
    def eri(*args):
        ERI = jnp.zeros((len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals)))
        for h, a in enumerate(atomic_orbitals):
            for i, b in enumerate(atomic_orbitals):
                for j, c in enumerate(atomic_orbitals):
                    for k, d in enumerate(atomic_orbitals):
                        if h <= i and j <= k:
                            eri_integral = generate_two_electron(a, b, c, d)(args[h], args[i], args[j], args[k])
                            ERI = jax.ops.index_update(ERI, jax.ops.index[(h, i, h, i), (i, h, i, h), (j, j, k, k), (k, k, j, j)], eri_integral)
        return ERI
    return eri


def kinetic_matrix(atomic_orbitals):
    """Computes the core Hamiltonian matrix"""
    def kinetic(*args):
        K = jnp.zeros((len(atomic_orbitals), len(atomic_orbitals)))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i <= j:
                    ham_integral = generate_kinetic(a, b)(args[i], args[j])
                    K = jax.ops.index_update(K, jax.ops.index[(i, j), (j, i)], ham_integral)
        return K
    return kinetic


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


def electron_nucleus_matrix(atomic_orbitals):
    """Computes the electron-nucleus interaction matrix"""
    def nuclear(*args):

        # Extracts nuclear coordinates
        C = []
        for count, atom in enumerate(atomic_orbitals):
            R, Coeff, A = build_param_space(atom.params, args[count])
            C.append(R)
        C = jnp.array(C)

        N = jnp.zeros((len(atomic_orbitals), len(atomic_orbitals)))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i <= j:
                    nuc_integral = 0
                    for c in C:
                        nuc_integral += generate_nuclear_attraction(a, b)(c, args[i], args[j])
                    N = jax.ops.index_update(N, jax.ops.index[(i, j), (j, i)], nuc_integral)
        return N
    return nuclear


def core_matrix(atomic_orbitals):
    """Computes the core matrix"""
    def core(*args):
        return -1 * electron_nucleus_matrix(atomic_orbitals)(*args) + kinetic_matrix(atomic_orbitals)(*args)
    return core


def exchange_matrix(coeffs, atomic_orbitals):
    """Computes the electron exchange matrix"""
    def exchange(*args):
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)
        P = density_matrix(coeffs)

        JM = jnp.einsum('pqrs,rs->pq', eri_tensor, P)
        KM = jnp.einsum('psqr,rs->pq', eri_tensor, P)
        return 2 * JM - KM
    return exchange


def fock_matrix(coeffs, atomic_orbitals):
    """Builds the Fock matrix"""
    def fock(*args):
        F = core_matrix(atomic_orbitals)(*args) + exchange_matrix(coeffs, atomic_orbitals)(*args)
        return F
    return fock


def hartree_fock(atomic_oribtals, tol=1e-8):
    """Performs the Hartree-Fock procedure"""
    def HF(*args):
        self_consistent = False

        F_initial = core_matrix(atomic_oribtals)(*args) # Builds the initial Fock matrix
        S = overlap_matrix(atomic_oribtals)(*args)
        v, w = jnp.linalg.eigh(S)
        v = jnp.array([1 / jnp.sqrt(r) for r in v])
        diag_mat, w_inv = jnp.diag(v), jnp.linalg.inv(w)
        X = w @ diag_mat @ w_inv

        F_tilde_initial = X.T @ F_initial @ X
        v_fock, w_fock = jnp.linalg.eigh(F_tilde_initial)
        coeffs = X @ w_fock

        print("Done Initial Iteration")
        counter = 0

        while not self_consistent:
            F = fock_matrix(coeffs, atomic_oribtals)(*args)
            F_tilde = X.T @ F @ X

            # Solve for eigenvalues and eigenvectors
            v_fock, w_fock = jnp.linalg.eigh(F_tilde)
            w_fock = X @ w_fock

            self_consistent = (jnp.linalg.norm(w_fock - coeffs) <= tol)
            coeffs = w_fock
            counter += 1
            print("Done Iteration: {}".format(counter))
        return v_fock, w_fock
    return HF
