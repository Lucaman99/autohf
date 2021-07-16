"""
Computing integrals relating to Hartree-Fock calculations
"""
import autograd.numpy as anp
import autograd.scipy as sc
from .utils import build_param_space, build_arr
import numpy as np
from autograd.extend import primitive, defvjp, defjvp


def double_factorial(n):
    """
    Computes the double factorial of n
    """
    k = 0
    prod = 1
    while n - k >= 0:
        prod = prod * (n - k)
        k = k + 2
    return prod


def gaussian_norm(L, alpha):
    """Normalizes some Gaussian"""
    l, m, n = L
    L_sum = l + m + n

    coeff = ((2 / anp.pi) ** 0.75) * ((2 ** L_sum) * (alpha ** (0.5 * L_sum + 0.75)))
    N = 1 / anp.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1))
    return coeff * N


def atomic_norm(L, alpha, a):
    """Normalizes some atomic orbital written as a sum of Gaussians"""
    l, m, n = L
    L_sum = l + m + n

    coeff = ((anp.pi ** (3/2)) / (2 ** L_sum))
    coeff = coeff * double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)

    s = ((a[:,anp.newaxis] * a) / ((alpha[:,anp.newaxis] + alpha) ** (L_sum + (3/2)))).sum()
    return 1 / anp.sqrt(coeff * s)


def expansion_coeff(i, j, t, Ra, Rb, alpha, beta):
    Qx = Ra - Rb
    p = alpha + beta
    q = (alpha * beta) / p
    c1, c2, c4 = (1 / (2 * p)), (q * Qx / alpha), (q * Qx / beta)

    pairs = [(i, j, t)]
    coeffs = [1.0]
    cs = 0.0

    while len(pairs) > 0:
        new_pairs = []
        new_coeffs = []
        for c, p in zip(coeffs, pairs):
            i, j, t = p
            c3 = t + 1
            if i == j == t == 0:
                cs = cs + c * anp.exp(-1 * q * (Qx ** 2))
            else:
                v1, v2, v3 = t - 1 >= 0, t <= (i + j - 1), t <= (i + j - 2)
                if j == 0:
                    if v1:
                        new_pairs.append((i - 1, j, t - 1))
                        new_coeffs.append(c * c1)
                    if v2:
                        new_pairs.append((i - 1, j, t))
                        new_coeffs.append(c * -c2)
                    if v3:
                        new_pairs.append((i - 1, j, t + 1))
                        new_coeffs.append(c * c3)
                else:
                    if v1:
                        new_pairs.append((i, j - 1, t - 1))
                        new_coeffs.append(c * c1)
                    if v2:
                        new_pairs.append((i, j - 1, t))
                        new_coeffs.append(c * c4)
                    if v3:
                        new_pairs.append((i, j - 1, t + 1))
                        new_coeffs.append(c * c3)
        coeffs = new_coeffs
        pairs = new_pairs
    return cs


def gaussian_overlap(alpha, L1, Ra, beta, L2, Rb):
    """
    Computes the overlap integral between two Gaussian functions
    """
    p = alpha + beta
    s = 1.0
    for i in range(3):
        s = s * anp.sqrt(anp.pi / p) * expansion_coeff(L1[i], L2[i], 0, Ra[i], Rb[i], alpha, beta)
    return s


def generate_overlap(a, b):
    """
    Computes a function that takes the free parameters of a and b
    as arguments, and computes the overlap integral.

    Structure of arguments is [R1, R2, C1, C2, A1, A2]
    """
    def S(*args):
        args_1, args_2 = args[0], args[1]
        R1, C1, A1 = build_param_space(a.params, args_1)
        R2, C2, A2 = build_param_space(b.params, args_2)

        C1 = C1 * gaussian_norm(a.L, A1)
        C2 = C2 * gaussian_norm(b.L, A2)
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        return N1 * N2 * ((C1[:,anp.newaxis] * C2) * gaussian_overlap(A1[:,anp.newaxis], a.L, R1, A2, b.L, R2)).sum()
    return S


def gaussian_kinetic(alpha, L1, Ra, beta, L2, Rb):
    """
    Computes the kinetic energy integral for primitive Gaussian functions
    """

    l1, m1, n1 = L1
    l2, m2, n2 = L2
    term0 = beta * (2 * (l2 + m2 + n2) + 3) * \
            gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2, m2, n2), Rb)
    term1 = -2 * (beta ** 2) * \
            (gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2 + 2, m2, n2), Rb) +
             gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2, m2 + 2, n2), Rb) +
             gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2, m2, n2 + 2), Rb))
    term2 = -0.5 * (l2 * (l2 - 1) * gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2 - 2, m2, n2), Rb) +
                    m2 * (m2 - 1) * gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2, m2 - 2, n2), Rb) +
                    n2 * (n2 - 1) * gaussian_overlap(alpha, (l1, m1, n1), Ra, beta, (l2, m2, n2 - 2), Rb))
    return term0 + term1 + term2


def generate_kinetic(a, b):
    """
    Computes the kinetic energy integral
    """
    def T(*args):
        args_1, args_2 = args[0], args[1]
        R1, C1, A1 = build_param_space(a.params, args_1)
        R2, C2, A2 = build_param_space(b.params, args_2)

        C1 = C1 * gaussian_norm(a.L, A1)
        C2 = C2 * gaussian_norm(b.L, A2)
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        return N1 * N2 * ((C1[:,anp.newaxis] * C2) * gaussian_kinetic(A1[:,anp.newaxis], a.L, R1, A2, b.L, R2)).sum()
    return T


def factorial(n):
    """Returns the factorial"""
    prod = 1
    for k in range(n):
        prod *= n - k
    return prod


def rising_factorial(n, lim):
    """Returns the rising factorial"""
    prod = 1
    for k in range(lim):
        prod *= n + k
    return prod


def boys_fn(n, t):
    return n * t

"""
@primitive
def boys_fn(n, t):
    val = anp.piecewise(t, [t == 0, t != 0], [lambda t : 1 / (2 * n + 1), lambda t : sc.special.gamma(0.5 + n) * sc.special.gammainc(0.5 + n, t) / (2 * (t ** (0.5 + n)))])
    return val


@primitive
def boys_fn_grad(n, t):
    val = anp.piecewise(t, [t == 0, t != 0], [lambda t : -1 / (2 * n + 3), lambda t : -1 * boys_fn(n + 1, t)])
    return val


defjvp(boys_fn_grad,
       None,
       lambda ans, n, t: lambda g: g * anp.piecewise(t, [t == 0, t != 0], [lambda t : 1 / (2 * n + 5), lambda t : boys_fn(n + 2, t)])
    )


defjvp(boys_fn,
       None,
       lambda ans, n, t: lambda g: g * boys_fn_grad(n, t)
    )
"""


def gaussian_prod(alpha, Ra, beta, Rb):
    """Returns the Gaussian product center"""
    return (alpha * anp.array(Ra) + beta * anp.array(Rb)) / (alpha + beta)


def R(t, u, v, n, p, DR):
    """Generates Hermite-Coulomb overlaps for nuclear attraction integral"""

    x, y, z = DR[0], DR[1], DR[2]

    T = p * (DR ** 2).sum(axis=0)
    val = 0
    if t == u == v == 0:
        val = val + ((-2 * p) ** n) * boys_fn(n, T)
    elif t == u == 0:
        if v > 1:
            val = val + (v - 1) * R(t, u, v - 2, n + 1, p, DR)
        val = val + z * R(t, u, v - 1, n + 1, p, DR)
    elif t == 0:
        if u > 1:
            val = val + (u - 1) * R(t, u - 2, v, n + 1, p, DR)
        val = val + y * R(t, u - 1, v, n + 1, p, DR)
    else:
        if t > 1:
            val = val + (t - 1) * R(t - 2, u, v, n + 1, p, DR)
        val = val + x * R(t - 1, u, v, n + 1, p, DR)
    return val


def nuclear_attraction(alpha, L1, Ra, beta, L2, Rb, C):
    """
    Computes nuclear attraction between Gaussian primitives
    Note that C is the coordinates of the nuclear centre
    """
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    p = alpha + beta
    P = gaussian_prod(alpha, Ra[:,anp.newaxis,anp.newaxis], beta, Rb[:,anp.newaxis,anp.newaxis])
    DR = P - anp.array(C)[:,anp.newaxis,anp.newaxis]

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val = val + expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) * \
                            expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) * \
                            expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) * \
                            R(t, u, v, 0, p, DR)
    val = val * 2 * anp.pi / p
    return val


def generate_nuclear_attraction(a, b):
    """
    Computes the nuclear attraction integral
    """
    def V(C, *args):
        args_1, args_2 = args[0], args[1]
        R1, C1, A1 = build_param_space(a.params, args_1)
        R2, C2, A2 = build_param_space(b.params, args_2)

        C1 = C1 * gaussian_norm(a.L, A1)
        C2 = C2 * gaussian_norm(b.L, A2)
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        val = N1 * N2 * ((C1 * C2[:,anp.newaxis]) * nuclear_attraction(A1, a.L, R1, A2[:,anp.newaxis], b.L, R2, C)).sum()
        return val
    return V


def electron_repulsion(alpha, L1, Ra, beta, L2, Rb, gamma, L3, Rc, delta, L4, Rd):
    """Electron repulsion between Gaussians"""
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    l3, m3, n3 = L3
    l4, m4, n4 = L4

    p = alpha + beta
    q = gamma + delta
    quotient = (p * q)/(p + q)

    P = gaussian_prod(alpha, Ra[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis], beta, Rb[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis]) # A and B composite center
    Q = gaussian_prod(gamma, Rc[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis], delta, Rd[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis]) # C and D composite center

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val = val + expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) * \
                                   expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) * \
                                   expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) * \
                                   expansion_coeff(l3, l4, tau, Rc[0], Rd[0], gamma, delta) * \
                                   expansion_coeff(m3, m4, nu, Rc[1], Rd[1], gamma, delta) * \
                                   expansion_coeff(n3, n4, phi, Rc[2], Rd[2], gamma, delta) * \
                                   ((-1) ** (tau + nu + phi)) * \
                                   R(t + tau, u + nu, v + phi, 0, quotient, P - Q)

    val = val * 2 * (anp.pi ** 2.5) / (p * q * anp.sqrt(p+q))
    return val


def generate_two_electron(a, b, c, d):
    """
    Computes the two electron repulsion integral
    """
    def EE(*args):
        args_1, args_2, args_3, args_4 = args[0], args[1], args[2], args[3]

        R1, C1, A1 = build_param_space(a.params, args_1)
        R2, C2, A2 = build_param_space(b.params, args_2)
        R3, C3, A3 = build_param_space(c.params, args_3)
        R4, C4, A4 = build_param_space(d.params, args_4)

        C1 = C1 * gaussian_norm(a.L, A1)
        C2 = C2 * gaussian_norm(b.L, A2)
        C3 = C3 * gaussian_norm(c.L, A3)
        C4 = C4 * gaussian_norm(d.L, A4)

        N1, N2, N3, N4 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2), atomic_norm(c.L, A3, C3), atomic_norm(d.L, A4, C4)

        val = N1 * N2 * N3 * N4 * (
                (C1 * C2[:,anp.newaxis] * C3[:,anp.newaxis,anp.newaxis] * C4[:,anp.newaxis,anp.newaxis,anp.newaxis]) *
                electron_repulsion(A1, a.L, R1, A2[:,anp.newaxis], b.L, R2, A3[:,anp.newaxis,anp.newaxis], c.L, R3, A4[:,anp.newaxis,anp.newaxis,anp.newaxis], d.L, R4)
        ).sum()
        return val
    return EE
