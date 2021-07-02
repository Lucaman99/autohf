"""Old code"""


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


@partial(custom_jvp, nondiff_argnums=(0, 1))
def hyp1f1(a, b, z):
    """Returns the hypergeometric function 1F1"""
    s = 1
    for t in range(1, lim + 1):
        s += (rising_factorial(a, t) / (rising_factorial(b, t) * factorial(t))) * (z ** t)
    return s

def density_matrix(num_elec, C):
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
    return jnp.dot(C[:,:num_elec//2],jnp.conjugate(C[:,:num_elec//2]).T)
