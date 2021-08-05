"""
Methods for performing Hartree-Fock
"""
from .integrals import *
import autograd.numpy as anp
from .utils import build_param_space, build_arr
from autograd.extend import primitive, defvjp, defjvp
import autograd


def overlap_matrix(atomic_orbitals):
    """
    Generates the overlap matrix
    """
    def overlap(*args):
        S = anp.eye(len(atomic_orbitals))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i < j:
                    overlap_integral = generate_overlap(a, b)(args[i], args[j])
                    S = S + overlap_integral * (build_arr(S.shape, (i, j)) + build_arr(S.shape, (j, i)))
        return S
    return overlap


def density_matrix(num_elec, C):
    """
    Computes the density matrix
    TODO: Understand this!
    TODO: Conjugate the second thing!
    """
    return anp.dot(C[:,:num_elec//2],anp.conj(C[:,:num_elec//2]).T)


def dict_ord(x, y):
    return x[1] <= y[1] if x[0] == y[0] else x[0] <= y[0]


def electron_repulsion_tensor(atomic_orbitals):
    """Computes a tensor of electron repulsion integrals"""
    def eri(*args):
        ERI = anp.zeros((len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals), len(atomic_orbitals)))
        for h, a in enumerate(atomic_orbitals):
            for i, b in enumerate(atomic_orbitals):
                for j, c in enumerate(atomic_orbitals):
                    for k, d in enumerate(atomic_orbitals):
                        if h <= i and j <= k and dict_ord((h, i), (j, k)):
                            eri_integral = generate_two_electron(a, b, c, d)(args[h], args[i], args[j], args[k])
                            mat = build_arr(ERI.shape, (h, i, j, k)) + \
                                  build_arr(ERI.shape, (i, h, j, k)) + \
                                  build_arr(ERI.shape, (h, i, k, j)) + \
                                  build_arr(ERI.shape, (i, h, k, j)) + \
                                  build_arr(ERI.shape, (j, k, h, i)) + \
                                  build_arr(ERI.shape, (j, k, i, h)) + \
                                  build_arr(ERI.shape, (k, j, h, i)) + \
                                  build_arr(ERI.shape, (k, j, i, h))
                            new_mat = anp.where(mat > 0, 1, 0)
                            ERI = ERI + eri_integral * new_mat
        return ERI
    return eri


def kinetic_matrix(atomic_orbitals):
    """Computes the core Hamiltonian matrix"""
    def kinetic(*args):
        K = anp.zeros((len(atomic_orbitals), len(atomic_orbitals)))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i <= j:
                    ham_integral = generate_kinetic(a, b)(args[i], args[j])
                    if i == j:
                        K = K + ham_integral * build_arr(K.shape, (i, j))
                    else:
                        K = K + ham_integral * (build_arr(K.shape, (i, j)) + build_arr(K.shape, (j, i)))
        return K
    return kinetic


def electron_nucleus_matrix(atomic_orbitals, charge):
    """Computes the electron-nucleus interaction matrix"""
    def nuclear(atom_R, *args):

        # Extracts nuclear coordinates FIX THIS!!!!!!
        C = atom_R
        N = anp.zeros((len(atomic_orbitals), len(atomic_orbitals)))
        for i, a in enumerate(atomic_orbitals):
            for j, b in enumerate(atomic_orbitals):
                if i <= j:
                    nuc_integral = 0
                    for k, c in enumerate(C):
                        nuc_integral = nuc_integral + charge[k] * generate_nuclear_attraction(a, b)(c, args[i], args[j])
                    if i == j:
                        N = N + nuc_integral * build_arr(N.shape, (i, j))
                    else:
                        N = N + nuc_integral * (build_arr(N.shape, (i, j)) + build_arr(N.shape, (j, i)))
        return N
    return nuclear


def core_matrix(charge, atomic_orbitals):
    """Computes the core matrix"""
    def core(atom_R, *args):
        return -1 * electron_nucleus_matrix(atomic_orbitals, charge)(atom_R, *args) + kinetic_matrix(atomic_orbitals)(*args)
    return core


def exchange_matrix(num_elec, coeffs, atomic_orbitals):
    """Computes the electron exchange matrix"""
    def exchange(*args):
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)
        P = density_matrix(num_elec, coeffs)

        JM = anp.einsum('pqrs,rs->pq', eri_tensor, P)
        KM = anp.einsum('psqr,rs->pq', eri_tensor, P)
        return 2 * JM - KM
    return exchange


def fock_matrix(num_elec, charge, coeffs, atomic_orbitals):
    """Builds the Fock matrix"""
    def fock(atom_R, *args):
        F = core_matrix(charge, atomic_orbitals)(atom_R, *args) + exchange_matrix(num_elec, coeffs, atomic_orbitals)(*args)
        return F
    return fock

"""
eigh_vec_fn = primitive(lambda a : anp.linalg.eigh(a)[1])
eigh_val_fn = primitive(lambda a : anp.linalg.eigh(a)[0])


def eigh_vec_grad(g, ans, x):
    return g @ autograd.jacobian(lambda a : anp.linalg.eigh(a)[1])(x)

def eigh_val_grad(g, ans, x):
    return g @ autograd.jacobian(lambda a : anp.linalg.eigh(a)[0])(x)

autograd.extend.defjvp(eigh_vec_fn, eigh_vec_grad)
autograd.extend.defjvp(eigh_val_fn, eigh_val_grad)
"""

@primitive
def eigh(M):
    return anp.linalg.eigh(M)


def _T(x): return anp.swapaxes(x, -1, -2)
def _H(x): return anp.conj(_T(x))
def symmetrize(x): return (x + _H(x)) / 2


def eigh_jvp(a_tangent, ans, a):
    epsilon = (1e-10)
    w, v = ans
    a_dot = a_tangent
    a_sym = symmetrize(a)
    w = w.astype(a.dtype)
    dot = anp.dot
    vdag_adot = dot(_H(v), a_dot)
    vdag_adot_v = dot(vdag_adot, v)

    deltas = w[..., anp.newaxis, :] - w[..., anp.newaxis]
    handle_degeneracies = True
    same_subspace = (abs(deltas) < epsilon
                     if handle_degeneracies
                     else anp.eye(a.shape[-1], dtype=bool))

    if handle_degeneracies:
        w_dot, v_dot = anp.linalg.eigh(vdag_adot_v * same_subspace)
        # Reorder these into sorted order of the original eigenvalues.
        # TODO(shoyer): consider rewriting with an explicit loop over degenerate
        # subspaces instead?
        v2 = dot(v, v_dot)
        w2 = anp.einsum('...ij,...jk,...ki->...i', _H(v2), a_sym, v2).real
        order = anp.argsort(w2, axis=-1)
        v = anp.take_along_axis(v2, order[..., anp.newaxis, :], axis=-1)
        dw = anp.take_along_axis(w_dot, order, axis=-1)
        deltas = w[..., anp.newaxis, :] - w[..., anp.newaxis]
        same_subspace = abs(deltas) < epsilon
    else:
        dw = anp.diagonal(vdag_adot_v, axis1=-2, axis2=-1)

    Fmat = anp.where(same_subspace, 0.0, 1.0 / deltas)
    C = Fmat * vdag_adot_v
    dv = dot(v, C)
    return dw, dv

defjvp(eigh, eigh_jvp)
norm = lambda x : anp.sqrt(anp.sum(anp.dot(x, x)))

def hartree_fock(num_elec, charge, atomic_orbitals, tol=1e-8):
    """Performs the Hartree-Fock procedure

    Note that this method does not necessarily build matrices using the methods
    constructed above.
    """

    def HF(atom_R, *args):
        self_consistent = False

        H_core = core_matrix(charge, atomic_orbitals)(atom_R, *args)  # Builds the initial Fock matrix
        S = overlap_matrix(atomic_orbitals)(*args)  # Builds the overlap matrix
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)  # Builds the electron repulsion tensor

        F_initial = H_core

        # Builds the X matrix
        v, w = eigh(S)
        v = anp.array([1 / anp.sqrt(r) for r in v])
        diag_mat, w_inv = anp.diag(v), w.T
        X = w @ diag_mat @ w_inv

        # Constructs F_tilde and finds the initial coefficients
        F_tilde_initial = X.T @ F_initial @ X
        v_fock, w_fock = eigh(F_tilde_initial)

        coeffs = X @ w_fock
        P = density_matrix(num_elec, coeffs)

        counter = 0
        F = F_tilde_initial
        """
        while not self_consistent:
            JM = anp.einsum('pqrs,rs->pq', eri_tensor, P)
            KM = anp.einsum('psqr,rs->pq', eri_tensor, P)
            E_mat = 2 * JM - KM

            F = H_core + E_mat
            F_tilde = anp.dot(X.T, anp.dot(F, X))

            # Solve for eigenvalues and eigenvectors
            v_fock, w_fock = eigh(F_tilde)
            w_fock = X @ w_fock
            P_new = density_matrix(num_elec, w_fock)

            self_consistent = (norm(P_new - P) <= tol)
            P = P_new

            counter += 1
        """
        return w, F, H_core, eri_tensor
    return HF