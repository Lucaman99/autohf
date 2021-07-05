# autohf
Automatic differentiation and Hartree-Fock

### 1. Background

Hartree-Fock is a method in computation quantum chemistry that treats the electrons in a molecule as a mean-field, and is able to find approximate solutions to the 
Schrodinger equation by taking linear combinations of atomic orbitals (LCAO).

### 2. Code Philosophy

The main idea behind AutoHF is a mapping from fundamental objects in Hartree-Fock, to maps from a parameter space to such objects. For example, instead of treating atomic oribtals as the fundamental
object used to perform calculations in Hartree-Fock, the fundamental objects are now maps from a parameter space to the space of atomic oribtals.

AutoHF is built on top of JAX, so we try to follow the general JAX-philosophy as much as possible.

### 3. To-Do

**Optimizations**

1. Using native JAX functionality to implement each of the recursive functions used to compute the integrals, so we can compile once with `jit`.


### Known Issues

- The method that computes electron-nucleus interactions assumes that the number of provided coordinate vectors is equal to the number of provided atomic basis functions. This is obviously not the case (it only really holds for hydrogen-based molecules).
