"""
Post-processing of Hartree-Fock results and generation of Hamiltonians
"""
import autograd.numpy as anp
from .hartreefock import *


def one_electron_integral(num_elec, atomic_orbitals, idx):
    """Returns the one electron coefficient for building the Hamiltonian"""
    def one_elec(*args):
        v_fock, w_fock, h_core, eri_tensor = hartree_fock(num_elec, atomic_orbitals)(*args)
        t = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)[idx[0]][idx[1]]
        return t
    return one_elec


def two_electron_integral(num_elec, atomic_orbitals, idx):
    """Returns two electron integral"""
    def two_elec(*args):
        v_fock, w_fock, h_core, eri_tensor = hartree_fock(num_elec, atomic_orbitals)(*args)
        t = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)[idx[0]][idx[1]][idx[2]][idx[3]]
        return t
    return two_elec


def electron_integrals(num_elec, atomic_orbitals):
    """Returns the one and two electron integrals"""
    def I(*args):
        v_fock, w_fock, h_core, eri_tensor = hartree_fock(num_elec, atomic_orbitals)(*args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        return one, two
    return I
