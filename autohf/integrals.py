"""
Computing integrals relating to Hartree-Fock calculations
"""
import jax.numpy as jnp
import jax.scipy as sc
from .utils import build_param_space, cartesian_prod
import jax
import numpy as np
from jax.experimental import loops


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

    with loops.Scope() as s:
        s.a = a
        s.alpha = alpha
        s.data = 0.0

        for i in s.range(s.a.shape[0]):
            for j in s.range(s.a.shape[0]):
                s.data += (s.a[i] * s.a[j]) / ((s.alpha[i] + s.alpha[j]) ** (L_sum + (3 / 2)))
        return 1 / jnp.sqrt(coeff * s.data)


def expansion_coeff(i, j, t, Ra, Rb, alpha, beta):
    """
    Computes expansion coefficients for calculation of overlap integrals
    """

    Qx = Ra - Rb
    p = alpha + beta
    q = (alpha * beta) / p

    return jax.lax.cond(
        jax.lax.bitwise_or((t < 0), (t > (i + j))),
        lambda _: 0.0,
        lambda _: jax.lax.cond(
            jax.lax.bitwise_and(jax.lax.bitwise_and(i == 0, j == 0), t == 0),
            lambda _: jnp.exp(-1 * q * (Qx ** 2)),
            lambda _: jax.lax.cond(
                j == 0,
                lambda _: (1 / (2 * p)) * expansion_coeff(i - 1, j, t - 1, Ra, Rb, alpha, beta) - \
                (q * Qx / alpha) * expansion_coeff(i - 1, j, t, Ra, Rb, alpha, beta) + \
                (t + 1) * expansion_coeff(i - 1, j, t + 1, Ra, Rb, alpha, beta),
                lambda _ : (1 / (2 * p)) * expansion_coeff(i, j - 1, t - 1, Ra, Rb, alpha, beta) + \
                (q * Qx / beta) * expansion_coeff(i, j - 1, t, Ra, Rb, alpha, beta) + \
                (t + 1) * expansion_coeff(i, j - 1, t + 1, Ra, Rb, alpha, beta),
                operand=None
            ),
            operand=None
        ),
        operand=None
    )


def gaussian_overlap(alpha, L1, Ra, beta, L2, Rb):
    """
    Computes the overlap integral between two Gaussian functions
    """
    p = alpha + beta
    integral = 1
    for j in range(3):
        integral = integral * jnp.sqrt(jnp.pi / p) * expansion_coeff(L1[j], L2[j], 0, Ra[j], Rb[j], alpha, beta)
    return integral


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

        C1 = [i * gaussian_norm(a.L, j) for i, j in zip(C1, A1)]
        C2 = [i * gaussian_norm(b.L, j) for i, j in zip(C2, A2)]
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        integral = 0
        for i, c1 in enumerate(C1):
            for j, c2 in enumerate(C2):
                integral = integral + N1 * N2 * c1 * c2 * gaussian_overlap(A1[i], a.L, R1, A2[j], b.L, R2)
        return integral
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

        C1 = [i * gaussian_norm(a.L, j) for i, j in zip(C1, A1)]
        C2 = [i * gaussian_norm(b.L, j) for i, j in zip(C2, A2)]
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        integral = 0
        for i, c1 in enumerate(C1):
            for j, c2 in enumerate(C2):
                integral = integral + N1 * N2 * c1 * c2 * gaussian_kinetic(A1[i], a.L, R1, A2[j], b.L, R2)
        return integral
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
    """Returns the Boys function"""
    return jax.lax.cond(
        t == 0,
        lambda r: 1 / (2 * r[0] + 1),
        lambda r: jax.lax.cond(
            n == 0,
            lambda r: jnp.sqrt(jnp.pi / (4 * r[1])) * sc.special.erf(jnp.sqrt(r[1])),
            lambda r: sc.special.gammaincc(0.5 + r[0], 0) * sc.special.gammainc(0.5 + r[0], r[1]) / (
                        (2 * r[1]) ** (0.5 + r[0])),
            r
        ),
        (n, t)
    )


def gaussian_prod(alpha, Ra, beta, Rb):
    """Returns the Gaussian product center"""
    return (alpha * jnp.array(Ra) + beta * jnp.array(Rb)) / (alpha + beta)


def R(t, u, v, n, p, DR):
    """Generates Hermite-Coulomb overlaps for nuclear attraction integral"""

    x, y, z = DR[0], DR[1], DR[2]

    T = p * (jnp.linalg.norm(DR) ** 2)
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
    P = gaussian_prod(alpha, Ra, beta, Rb)
    DR = P - jnp.array(C)

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val = val + expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) * \
                            expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) * \
                            expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) * \
                            R(t, u, v, 0, p, DR)
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

        C1 = [i * gaussian_norm(a.L, j) for i, j in zip(C1, A1)]
        C2 = [i * gaussian_norm(b.L, j) for i, j in zip(C2, A2)]
        N1, N2 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2)

        integral = 0
        for i, c1 in enumerate(C1):
            for j, c2 in enumerate(C2):
                integral = integral + N1 * N2 * c1 * c2 * nuclear_attraction(A1[i], a.L, R1, A2[j], b.L, R2, C)
        return integral

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

    P = gaussian_prod(alpha, Ra, beta, Rb) # A and B composite center
    Q = gaussian_prod(gamma, Rc, delta, Rd) # C and D composite center

    with loops.Scope() as s:
        s.val = 0.0
        for t in s.range(l1+l2+1):
            for u in s.range(m1+m2+1):
                for v in s.range(n1+n2+1):
                    for tau in s.range(l3+l4+1):
                        for nu in s.range(m3+m4+1):
                            for phi in s.range(n3+n4+1):
                                s.val += expansion_coeff(l1, l2, t, Ra[0], Rb[0], alpha, beta) * \
                                       expansion_coeff(m1, m2, u, Ra[1], Rb[1], alpha, beta) * \
                                       expansion_coeff(n1, n2, v, Ra[2], Rb[2], alpha, beta) * \
                                       expansion_coeff(l3, l4, tau, Rc[0], Rd[0], gamma, delta) * \
                                       expansion_coeff(m3, m4, nu, Rc[1], Rd[1], gamma, delta) * \
                                       expansion_coeff(n3, n4, phi, Rc[2], Rd[2], gamma, delta) * \
                                       ((-1) ** tau + nu + phi) * \
                                       R(t + tau, u + nu, v + phi, 0, quotient, P - Q)

        return s.val * 2 * (jnp.pi ** 2.5) / (p * q * jnp.sqrt(p+q))


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

        C1 = [i * gaussian_norm(a.L, j) for i, j in zip(C1, A1)]
        C2 = [i * gaussian_norm(b.L, j) for i, j in zip(C2, A2)]
        C3 = [i * gaussian_norm(c.L, j) for i, j in zip(C3, A3)]
        C4 = [i * gaussian_norm(d.L, j) for i, j in zip(C4, A4)]

        N1, N2, N3, N4 = atomic_norm(a.L, A1, C1), atomic_norm(b.L, A2, C2), atomic_norm(c.L, A3, C3), atomic_norm(d.L, A4, C4)

        integral = 0
        for h, c1 in enumerate(C1):
            for i, c2 in enumerate(C2):
                for j, c3 in enumerate(C3):
                    for k, c4 in enumerate(C4):
                        integral = integral + N1 * N2 * N3 * N4 * c1 * c2 * c3 * c4 * electron_repulsion(A1[h], a.L, R1, A2[i], b.L, R2, A3[j], c.L, R3, A4[k], d.L, R4)
        return integral
    return EE
