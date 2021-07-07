"""
Methods for performing Hartree-Fock
"""
from .integrals import *
import jax.numpy as jnp
from .utils import build_param_space, build_arr


def overlap_matrix(atomic_orbitals):
    """
    Generates the overlap matrix
    """
    def overlap(*args):
        S = jnp.eye(len(atomic_orbitals))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i < j:
                    overlap_integral = generate_overlap(a, b)(args[i], args[j])
                    S = jax.ops.index_update(S, ([i, j], [j, i]), overlap_integral)
        return S
    return overlap


def density_matrix(num_elec, C):
    """
    Computes the density matrix
    TODO: Understand this!
    """
    return jnp.dot(C[:,:num_elec//2],jnp.conjugate(C[:,:num_elec//2]).T)


def dict_ord(x, y):
    return x[1] <= y[1] if x[0] == y[0] else x[0] <= y[0]


def electron_repulsion_tensor(atomic_orbitals):
    """Computes a tensor of electron repulsion integrals"""

    def eri(*args):
        ERI = jnp.zeros((len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals)))

        for h, a in enumerate(atomic_orbitals):
            for i, b in enumerate(atomic_orbitals):
                for j, c in enumerate(atomic_orbitals):
                    for k, d in enumerate(atomic_orbitals):
                        if h <= i and j <= k and dict_ord((h, i), (j, k)):
                            eri_integral = generate_two_electron(a, b, c, d)(args[h], args[i], args[j], args[k])
                            ERI = jax.ops.index_update(ERI, ([h, h, i, i, j, j, k, k], [i, i, h, h, k, k, j, j], [j, k, j, k, h, i, h, i], [k, j, k, j, i, h, i, h]), eri_integral)
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
                    K = jax.ops.index_update(K, ([i, j], [j, i]), ham_integral)
        return K
    return kinetic


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
                        nuc_integral = nuc_integral + generate_nuclear_attraction(a, b)(c, args[i], args[j])
                        N = jax.ops.index_update(N, ([i, j], [j, i]), nuc_integral)
        return N
    return nuclear


def core_matrix(atomic_orbitals):
    """Computes the core matrix"""
    def core(*args):
        return -1 * electron_nucleus_matrix(atomic_orbitals)(*args) + kinetic_matrix(atomic_orbitals)(*args)
    return core


def exchange_matrix(num_elec, coeffs, atomic_orbitals):
    """Computes the electron exchange matrix"""
    def exchange(*args):
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)
        P = density_matrix(num_elec, coeffs)

        JM = jnp.einsum('pqrs,rs->pq', eri_tensor, P)
        KM = jnp.einsum('psqr,rs->pq', eri_tensor, P)
        return 2 * JM - KM
    return exchange


def fock_matrix(num_elec, coeffs, atomic_orbitals):
    """Builds the Fock matrix"""
    def fock(*args):
        F = core_matrix(atomic_orbitals)(*args) + exchange_matrix(num_elec, coeffs, atomic_orbitals)(*args)
        return F
    return fock


def hartree_fock(num_elec, atomic_orbitals, tol=1e-8):
    """Performs the Hartree-Fock procedure

    Note that this method does not necessarily build matrices using the methods
    constructed above.
    """
    def HF(*args):
        self_consistent = False

        H_core = core_matrix(atomic_orbitals)(*args) # Builds the initial Fock matrix
        S = overlap_matrix(atomic_orbitals)(*args) # Builds the overlap matrix
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args) # Builds the electron repulsion tensor

        F_initial = H_core

        # Builds the X matrix
        v, w = jnp.linalg.eigh(S)
        v = jnp.array([1 / jnp.sqrt(r) for r in v])
        diag_mat, w_inv = jnp.diag(v), w.T
        X = w @ diag_mat @ w_inv

        # Constructs F_tilde and finds the initial coefficients
        F_tilde_initial = X.T @ F_initial @ X
        v_fock, w_fock = jnp.linalg.eigh(F_tilde_initial)

        coeffs = X @ w_fock
        P = density_matrix(num_elec, coeffs)

        counter = 0
        while not self_consistent:

            JM = jnp.einsum('pqrs,rs->pq', eri_tensor, P)
            KM = jnp.einsum('psqr,rs->pq', eri_tensor, P)
            E_mat = 2 * JM - KM

            F = H_core + E_mat
            F_tilde = X.T @ F @ X

            # Solve for eigenvalues and eigenvectors
            v_fock, w_fock = jnp.linalg.eigh(F_tilde)
            w_fock = X @ w_fock
            P_new = density_matrix(num_elec, w_fock)

            self_consistent = (jnp.linalg.norm(P_new - P) <= tol)
            P = P_new

            counter += 1
        return v_fock, w_fock, H_core, eri_tensor
    return HF
