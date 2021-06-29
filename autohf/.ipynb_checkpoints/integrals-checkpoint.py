"""
Computing integrals relating to Hartree-Fock calculations
"""
import jax.numpy as jnp


def double_factorial(n):
    """
    Computes the double factorial of n
    """
    k = 0
    prod = 1
    while n - k >= 0:
        prod *= n - k
        k += 2
    return prod


def gaussian_norm(L, alpha):
    """Normalizes some Gaussian"""
    l, m, n = L
    L_sum = l + m + n

    coeff = ((2 / jnp.pi) ** 0.75) * ((2 ** L_sum) * (alpha ** (0.5 * L + 0.75)))
    N = 1 / jnp.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1))
    return jnp.sqrt(coeff * N)


def atomic_norm(L, alpha, a):
    """Normalizes some atomic orbital written as a sum of Gaussians"""
    l, m, n = L
    L_sum = l + m + n

    coeff = ((jnp.pi ** (3/2)) / (2 ** L_sum))
    coeff = coeff * double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)

    s = 0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            s += (a[i] * a[j]) / ((alpha[i] + alpha[j]) ** (L_sum + (3/2)))

    return 1 / jnp.sqrt(coeff * s)


def expansion_coeff(i, j, t, Ra, Rb, alpha, beta):
    """
    Computes expansion coefficients for calculation of overlap integrals
    """

    Qx = Ra - Rb
    p = alpha + beta
    q = (alpha * beta) / p

    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return jnp.exp(-1 * q * (Qx ** 2))
    elif j == 0:
        return (1 / (2 * p)) * expansion_coeff(i - 1, j, t - 1, Ra, Rb, alpha, beta) - \
               (q * Qx / alpha) * expansion_coeff(i - 1, j, t, Ra, Rb, alpha, beta) + \
               (t + 1) * expansion_coeff(i - 1, j, t + 1, Ra, Rb, alpha, beta)
    else:
        return (1 / (2 * p)) * expansion_coeff(i, j - 1, t - 1, Ra, Rb, alpha, beta) + \
               (q * Qx / beta) * expansion_coeff(i, j - 1, t, Ra, Rb, alpha, beta) + \
               (t + 1) * expansion_coeff(i, j - 1, t + 1, Ra, Rb, alpha, beta)


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

        integral = 0
        for i, c1 in enumerate(C1):
            for j, c2 in enumerate(C2):
                integral += c1 * c2 * gaussian_overlap(A1[i], a.L, R1, A2[j], b.L, R2)
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

        integral = 0
        for i, c1 in enumerate(C1):
            for j, c2 in enumerate(C2):
                integral += c1 * c2 * gaussian_kinetic(A1[i], a.L, R1, A2[j], b.L, R2)
        return integral
    return T


def factorial(n):
    prod = 1
    for k in range(n):
        prod *= n - k
    return prod


def rising_factorial(n, lim):
    prod = 1
    for k in range(lim):
        prod *= n + k
    return prod


def hyp1f1(a, b, z, lim=500):
    s = 1
    for t in range(1, lim + 1):
        s += (rising_factorial(a, t) / (rising_factorial(b, t) * factorial(t))) * (z ** t)
    return s


def boys_fn(n, t):
    return hyp1f1(n + 0.5, n + 1.5, -1 * t) / (2 * n + 1)


def hermite_coulomb():
    pass


def generate_nuclear_attraction(a, b):
    """
    Computes the nuclear attraction integral
    """
    def V(*args):
        pass


def generate_two_electron(a, b):
    """
    Computes the two electron repulsion integral
    """
    def EE(*args):
        pass
