"""
Methods for performing Hartree-Fock
"""
from importlib_metadata import itertools
from .integrals import *
import autograd.numpy as anp
from .utils import build_param_space, build_arr
from autograd.extend import primitive, defvjp, defjvp
import autograd
import algopy
from algopy import UTPM
import itertools


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


def holomorphic_density_matrix(num_elec, C):
    """
    Computes the density matrix
    TODO: Understand this!
    TODO: Conjugate the second thing!
    """
    return anp.dot(C[:,:num_elec//2],C[:,:num_elec//2].T)

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
    v, w = anp.linalg.eigh(M)
    return v, w

@primitive
def eig(M):
    v, w = anp.linalg.eig(M)
    return v, w

def norm_eigh(M):
    v, w = eigh(M)
    new_w = anp.where(w[0] <= 0, -w, w)
    return v, new_w

def norm_eig(M):
    v, w = eig(M)
    new_w = anp.where(w[0] <= 0, -w, w)
    return v, new_w

def _T(x): return anp.swapaxes(x, -1, -2)
def _H(x): return anp.conj(_T(x))
def symmetrize(x): return (x + _H(x)) / 2


def take_along_axis(arr, ind, axis=0):
    if axis < 0:
       if axis >= -arr.ndim:
           axis += arr.ndim
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (arr.ndim - 1)   #inserted dimensions

    dest_dims = list(range(axis)) + [None] + list(range(axis+ins_ndim, ind.ndim))

    # could also call np.ix_ here with some dummy arguments, then throw those results away
    inds = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim+1:]
            inds.append(anp.arange(n).reshape(ind_shape_dim))
    return arr[tuple(inds)]


def eigh_jvp(a_tangent, ans, a):
    epsilon = (1e-10)
    w, v = ans
    a_dot = a_tangent
    a_sym = symmetrize(a)
    #w = w.astype(a.dtype)
    dot = anp.dot
    vdag_adot = dot(_H(v), a_dot)
    vdag_adot_v = dot(vdag_adot, v)

    deltas = w[..., anp.newaxis, :] - w[..., anp.newaxis]
    handle_degeneracies = True
    same_subspace = (abs(deltas) < epsilon
                     if handle_degeneracies
                     else anp.eye(a.shape[-1], dtype=bool))

    if handle_degeneracies:
        w_dot, v_dot = eigh(vdag_adot_v * same_subspace)
        # Reorder these into sorted order of the original eigenvalues.
        # TODO(shoyer): consider rewriting with an explicit loop over degenerate
        # subspaces instead?
        v2 = dot(v, v_dot)
        w2 = anp.real(anp.einsum('...ij,...jk,...ki->...i', _H(v2), a_sym, v2))
        order = anp.argsort(w2, axis=-1)
        v = take_along_axis(v2, order[..., anp.newaxis, :], axis=-1)
        dw = take_along_axis(w_dot, order, axis=-1)
        deltas = w[..., anp.newaxis, :] - w[..., anp.newaxis]
        same_subspace = abs(deltas) < epsilon
    else:
        dw = anp.diagonal(vdag_adot_v, axis1=-2, axis2=-1)

    Fmat = anp.where(same_subspace, 0.0, 1.0 / deltas)
    C = Fmat * vdag_adot_v
    dv = dot(v, C)
    return dw, dv

defjvp(eigh, eigh_jvp)

@primitive
def cracked_eigh(M):
    v, w = anp.linalg.eigh(M)
    return v, w

@primitive
def cracked_eigval(a):
    dw, dv = algopy.eigh(a)
    return dw

@primitive
def cracked_eigvec(a):
    dw, dv = algopy.eigh(a)
    return dv

@primitive
def c_eigval_jacobian(a):
    """Eigenvalue Jacobian"""
    x = UTPM.init_jacobian(a)
    y = cracked_eigval(x)
    algopy_jacobian = UTPM.extract_jacobian(y)
    return algopy_jacobian

@primitive
def c_eigvec_jacobian(a):
    """Eigenvector Jacobian"""
    x = UTPM.init_jacobian(a)
    y = cracked_eigvec(x)
    algopy_jacobian = UTPM.extract_jacobian(y)
    return algopy_jacobian


def c_hvp_eigval(a_tangent, ans, a):
    """Hessian vector product --> jvp of eigval jacobian"""
    x = UTPM.init_hessian(a)
    y = cracked_eigval(x.reshape(a.shape))
    H = np.zeros((a.shape[0], a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        H[i] = UTPM.extract_hessian(a.shape[0], y[i])
    out = H.reshape((a.shape[0], a.shape[0] ** 2)) @ a_tangent.flatten()
    return out


def c_hvp_eigvec(a_tangent, ans, a):
    """Hessian vector product --> jvp of eigvec jacobian"""
    x = UTPM.init_hessian(a)
    y = cracked_eigvec(x.reshape(a.shape))
    H = np.zeros((a.shape[0], a.shape[0], a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            H[i][j] = UTPM.extract_hessian(a.shape[0], y[i][j])

    return np.dot(H.reshape((a.shape[0], a.shape[0], a.shape[0] ** 2)), a_tangent.flatten()).reshape(a.shape)


def c_eigval_jvp(a_tangent, ans, a):
    out = anp.array(c_eigval_jacobian(a) @ a_tangent.flatten())
    return out


def c_eigvec_jvp(a_tangent, ans, a):
    return anp.array(c_eigvec_jacobian(a) @ a_tangent.flatten()).reshape(a.shape)

defjvp(cracked_eigval, c_eigval_jvp)
defjvp(cracked_eigvec, c_eigvec_jvp)
defjvp(c_eigval_jacobian, c_hvp_eigval)
defjvp(c_eigvec_jacobian, c_hvp_eigvec)

norm = lambda x : anp.sqrt(anp.sum(anp.dot(x, x)))


def hartree_fock(num_elec, charge, atomic_orbitals, guess=None, tol=1e-8):
    """Performs the Hartree-Fock procedure

    Note that this method does not necessarily build matrices using the methods
    constructed above.
    """

    def HF(atom_R, *args):
        self_consistent = False

        H_core = core_matrix(charge, atomic_orbitals)(atom_R, *args)  # Builds the initial Fock matrix
        K = kinetic_matrix(atomic_orbitals)(*args)
        V = -1 * electron_nucleus_matrix(atomic_orbitals, charge)(atom_R, *args)
        S = overlap_matrix(atomic_orbitals)(*args)  # Builds the overlap matrix
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)  # Builds the electron repulsion tensor

        F_initial = H_core

        # Fudge factor!
        convergence = 1e-8
        fudge = anp.array(np.linspace(0, 1, S.shape[0])) * convergence
        shift = anp.diag(fudge)

        S = S + shift

        v, w = norm_eigh(S)
        v = anp.array([1 / anp.sqrt(r) for r in v])
        diag_mat, w_inv = anp.diag(v), w.T
        X = w @ diag_mat @ w_inv

        # Constructs F_tilde and finds the initial coefficients
        F_tilde_initial = X.T @ F_initial @ X
        F_tilde_initial = F_tilde_initial + shift
        v_fock, w_fock = norm_eigh(F_tilde_initial)

        coeffs = X @ w_fock
        P = density_matrix(num_elec, coeffs)

        counter = 0
        for _ in range(50):
            JM = anp.einsum('pqrs,rs->pq', eri_tensor, P)
            KM = anp.einsum('psqr,rs->pq', eri_tensor, P)
            E_mat = 2 * JM - KM

            F = H_core + E_mat
            F_tilde = anp.dot(X.T, anp.dot(F, X))
            F_tilde = F_tilde + shift

            # Solve for eigenvalues and eigenvectors
            v_fock, w_fock = norm_eigh(F_tilde)
            w_fock = X @ w_fock

            P_new = density_matrix(num_elec, w_fock)

            self_consistent = (norm(P_new - P) <= tol)
            P = P_new

            counter += 1
        
        if guess is not None:

            # Enforces ordering convention
            perms = list(itertools.permutations(list(range(len(w_fock)))))
            scores = []
            for p in perms:
                scores.append(((anp.abs(w_fock)[:,p] - anp.abs(guess)) ** 2).sum())
            w_fock = w_fock[:,perms[scores.index(min(scores))]]

            # Enforces sign convention
            w_new = []
            for c, row in enumerate(w_fock.T):
                score = ((row - guess.T[c]) ** 2).sum() < (((-1 * row) - guess.T[c]) ** 2).sum()
                if score:
                    w_new.append(row)
                else:
                    w_new.append(-1 * row)
            w_fock = anp.array(w_new).T
        
        return w_fock, F, H_core, eri_tensor, K, V, S
    return HF

"""
def holomorphic_hartree_fock(num_elec, charge, atomic_orbitals, initial=None, tol=1e-8):

    def HF(atom_R, *args):
        self_consistent = False

        H_core = core_matrix(charge, atomic_orbitals)(atom_R, *args)  # Builds the initial Fock matrix
        K = kinetic_matrix(atomic_orbitals)(*args)
        V = -1 * electron_nucleus_matrix(atomic_orbitals, charge)(atom_R, *args)
        S = overlap_matrix(atomic_orbitals)(*args)  # Builds the overlap matrix
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)  # Builds the electron repulsion tensor

        F_initial = initial if initial is not None else H_core

        # Fudge factor!
        convergence = 1e-8
        fudge = anp.array(np.linspace(0, 1, S.shape[0])) * convergence
        shift = anp.diag(fudge)

        S = S + shift

        v, w = norm_eig(S)
        v = anp.array([1 / anp.sqrt(r) for r in v])
        diag_mat, w_inv = anp.diag(v), w.T
        X = w @ diag_mat @ w_inv

        # Constructs F_tilde and finds the initial coefficients
        F_tilde_initial = X.T @ F_initial @ X
        F_tilde_initial = F_tilde_initial + shift
        v_fock, w_fock = norm_eig(F_tilde_initial)

        coeffs = X @ w_fock
        P = holomorphic_density_matrix(num_elec, coeffs)

        counter = 0
        while not self_consistent:
            JM = anp.einsum('pqrs,rs->pq', eri_tensor, P)
            KM = anp.einsum('psqr,rs->pq', eri_tensor, P)
            E_mat = 2 * JM - KM

            F = H_core + E_mat
            F_tilde = anp.dot(X.T, anp.dot(F, X))
            F_tilde = F_tilde + shift

            # Solve for eigenvalues and eigenvectors
            v_fock, w_fock = norm_eig(F_tilde)
            w_fock = X @ w_fock
            P_new = density_matrix(num_elec, w_fock)

            self_consistent = (norm(P_new - P) <= tol)
            P = P_new

            counter += 1
        return w_fock, F, H_core, eri_tensor, K, V, S
    return HF
"""


def grad_hartree_fock(num_elec, charge, atomic_orbitals, guess=None, tol=1e-8):
    """Performs the Hartree-Fock procedure with gradient descent

    Note that this method does not necessarily build matrices using the methods
    constructed above.
    
    anp.einsum('pq,qp', fock + h_core, density_matrix(num_elec, w_fock))
    """

    def HF_energy(atom_R, *args):
        S = overlap_matrix(atomic_orbitals)(*args)  # Builds the overlap matrix
        H_core = core_matrix(charge, atomic_orbitals)(atom_R, *args)  # Builds the initial Fock matrix
            
        eri_tensor = electron_repulsion_tensor(atomic_orbitals)(*args)  # Builds the electron repulsion tensor

        # Fudge factor!
        convergence = 1e-8
        fudge = anp.array(np.linspace(0, 1, S.shape[0])) * convergence
        shift = anp.diag(fudge)

        S = S + shift

        def fn(coeffs):
            P = density_matrix(num_elec, coeffs)

            JM = anp.einsum('pqrs,rs->pq', eri_tensor, P)
            KM = anp.einsum('psqr,rs->pq', eri_tensor, P)
            E_mat = 2 * JM - KM

            F = H_core + E_mat
            hf_energy = anp.einsum('pq,qp', F + H_core, P) # Computes HF energy
            overlap_val = anp.einsum('pq,pr,qs', S, coeffs, coeffs) # Computes the overlap value
            return hf_energy, overlap_val
        return fn, S
    return HF_energy