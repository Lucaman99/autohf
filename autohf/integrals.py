"""
Computing integrals relating to Hartree-Fock calculations
"""
import jax.numpy as jnp
import jax.scipy as sc
from .utils import build_param_space, build_arr
import numpy as np
import jax
from jax import custom_jvp
from functools import partial


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

    coeff = ((2 / jnp.pi) ** 0.75) * ((2 ** L_sum) * (alpha ** (0.5 * L_sum + 0.75)))
    N = 1 / jnp.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1))
    return coeff * N


def atomic_norm(L, alpha, a):
    """Normalizes some atomic orbital written as a sum of Gaussians"""
    l, m, n = L
    L_sum = l + m + n

    coeff = ((jnp.pi ** (3/2)) / (2 ** L_sum))
    coeff = coeff * double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)

    s = ((a[:,jnp.newaxis] * a) / ((alpha[:,jnp.newaxis] + alpha) ** (L_sum + (3/2)))).sum()
    return 1 / jnp.sqrt(coeff * s)


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
                cs = cs + c * jnp.exp(-1 * q * (Qx ** 2))
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
        s = s * jnp.sqrt(jnp.pi / p) * expansion_coeff(L1[i], L2[i], 0, Ra[i], Rb[i], alpha, beta)
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

        return N1 * N2 * ((C1[:,jnp.newaxis] * C2) * gaussian_overlap(A1[:,jnp.newaxis], a.L, R1, A2, b.L, R2)).sum()
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

        return N1 * N2 * ((C1[:,jnp.newaxis] * C2) * gaussian_kinetic(A1[:,jnp.newaxis], a.L, R1, A2, b.L, R2)).sum()
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

import scipy

@partial(custom_jvp, nondiff_argnums=(0,))
def boys_fn(n, t):
    val = jnp.where(t == 0, 1 / (2 * n + 1), scipy.special.gamma(0.5 + n) * sc.special.gammainc(0.5 + n, t) / (2 * (t ** (0.5 + n))))
    return val


@partial(custom_jvp, nondiff_argnums=(0,))
def boys_fn_grad(n, t):
    val = jnp.where(t == 0, -1 / (2 * n + 3), -1 * boys_fn(n + 1, t))
    return val


@boys_fn_grad.defjvp
def boys_fn_grad_jvp(n, primals, tangents):
  t, = primals
  t_dot, = tangents
  return boys_fn_grad(n, t), jnp.where(t == 0, 1 / (2 * n + 5), boys_fn(n + 2, t)) * t_dot


@boys_fn.defjvp
def boys_fn_jvp(n, primals, tangents):
    t, = primals
    t_dot, = tangents
    return boys_fn(n, t), boys_fn_grad(n, t) * t_dot


def gaussian_prod(alpha, Ra, beta, Rb):
    """Returns the Gaussian product center"""
    return (alpha * jnp.array(Ra) + beta * jnp.array(Rb)) / (alpha + beta)


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
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    p = alpha + beta
    P = gaussian_prod(alpha, Ra[:,jnp.newaxis,jnp.newaxis], beta, Rb[:,jnp.newaxis,jnp.newaxis])
    DR = P - jnp.array(C)[:,jnp.newaxis,jnp.newaxis]

    e1 = jnp.array([expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) for t in range(l1 + l2 + 1)])
    e2 = jnp.array([expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) for u in range(m1 + m2 + 1)])
    e3 = jnp.array([expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) for v in range(n1 + n2 + 1)])

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val = val + e1[t] * e2[u] * e3[v] * R(t, u, v, 0, p, DR)
    val = val * 2 * jnp.pi / p
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

        val = N1 * N2 * ((C1 * C2[:,jnp.newaxis]) * nuclear_attraction(A1, a.L, R1, A2[:,jnp.newaxis], b.L, R2, C)).sum()
        return val
    return V


def electron_repulsion(alpha, L1, Ra, beta, L2, Rb, gamma, L3, Rc, delta, L4, Rd):
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    l3, m3, n3 = L3
    l4, m4, n4 = L4

    p = alpha + beta
    q = gamma + delta
    quotient = (p * q)/(p + q)

    P = gaussian_prod(alpha, Ra[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis], beta,
                      Rb[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis]) # A and B composite center
    Q = gaussian_prod(gamma, Rc[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis], delta,
                      Rd[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis]) # C and D composite center


    e1 = jnp.array([expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) for t in range(l1+l2+1)])
    e2 = jnp.array([expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) for u in range(m1+m2+1)])
    e3 = jnp.array([expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) for v in range(n1+n2+1)])
    e4 = jnp.array([((-1) ** tau) * expansion_coeff(l3, l4, tau, Rc[0], Rd[0], gamma, delta) for tau in range(l3+l4+1)])
    e5 = jnp.array([((-1) ** nu) * expansion_coeff(m3, m4, nu, Rc[1], Rd[1], gamma, delta) for nu in range(m3+m4+1)])
    e6 = jnp.array([((-1) ** phi) * expansion_coeff(n3, n4, phi, Rc[2], Rd[2], gamma, delta) for phi in range(n3+n4+1)])

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val = val + e1[t] * e2[u] * e3[v] * e4[tau] * e5[nu] * e6[phi] * R(t + tau, u + nu, v + phi, 0, quotient, P - Q)

    val = val * 2 * (jnp.pi ** 2.5) / (p * q * jnp.sqrt(p+q))
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
                (C1 * C2[:,jnp.newaxis] * C3[:,jnp.newaxis,jnp.newaxis] * C4[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]) *
                electron_repulsion(A1, a.L, R1, A2[:,jnp.newaxis], b.L, R2, A3[:,jnp.newaxis,jnp.newaxis], c.L, R3,
                                   A4[:,jnp.newaxis,jnp.newaxis,jnp.newaxis], d.L, R4)
        ).sum()
        return val
    return EE
