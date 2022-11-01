"""
Post-processing of Hartree-Fock results and generation of Hamiltonians
"""
import autograd.numpy as anp
from .hartreefock import *
from .utils import cartesian_prod, build_arr
import openfermion
from pennylane import qchem


def electron_integrals(num_elec, charge, atomic_orbitals):
    """Returns the one and two electron integrals"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        return one, two
    return I

def electron_integrals_flat(num_elec, charge, atomic_orbitals, guess=None, occupied=None, active=None):
    """Returns the one and two electron integrals flattened into a 1D array"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals, guess=guess)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
        return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten()))
    return I

def electron_integrals_flat_new(num_elec, charge, atomic_orbitals, initial=None, occupied=None, active=None):
    """Returns the one and two electron integrals flattened into a 1D array"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals, initial=initial)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
        return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten())), w_fock
    return I

def hf_coeffs(num_elec, charge, atomic_orbitals, guess=None, occupied=None, active=None):
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals, guess=guess)(atom_R, *args)
        return w_fock
    return I

def electron_integrals_flat_w(num_elec, charge, atomic_orbitals, w, occupied=None, active=None):
    """Returns the one and two electron integrals flattened into a 1D array"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w.T, h_core, w)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w.T, w.T, eri_tensor, w, w), 1, 3)
        core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
        return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten()))
    return I

def universal_electron_integrals_flat(num_elec, charge, atomic_orbitals, occupied=None, active=None):
    """Returns the electron integrals needed to approximate the universal functional"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, K, w_fock)
        one_v = anp.einsum("qr,rs,st->qt", w_fock.T, V, w_fock)
        one_s = anp.einsum("qr,rs,st->qt", w_fock.T, S, w_fock)

        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)

        core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
        core_fake, one_elec_v, two_elec = get_active_space_integrals(one_v, two, occupied_indices=occupied, active_indices=active)
        core_fake, one_elec_s, two_elec = get_active_space_integrals(one_s, two, occupied_indices=occupied, active_indices=active)

        return anp.concatenate((anp.array([core]), one_elec.flatten(), one_elec_v.flatten(), one_elec_s.flatten(), two_elec.flatten()))
    return I

def one_body_potential_integrals(num_elec, charge, atomic_orbitals):
    """Computes the one-body potential energy integrals"""
    def I(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        mat = electron_nucleus_matrix(atomic_orbitals, charge)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, mat, w_fock)
        return one
    return I

def electron_integrals_known(one, two, occupied=None, active=None):
    """Returns the one and two electron integrals flattened into a 1D array"""
    core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
    return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten()))


def distance(pair):
    return anp.sqrt(((pair[0] - pair[1]) ** 2).sum())


def nuclear_energy(charges):
    """
    Generates the repulsion between nuclei of the atoms in a molecule
    """
    def repulsion(R):
        R = R.reshape((len(charges), 3))
        s = 0
        for i, r1 in enumerate(R):
            for j, r2 in enumerate(R):
                if i > j:
                    s = s + (charges[i] * charges[j] / distance([r1, r2]))
        return s
    return repulsion


def build_h_from_integrals(geometry, one_electron, two_electron, nuc_energy, wires, basis="sto-3g", multiplicity=1,
                           charge=0):
    molecule = openfermion.MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
    molecule.one_body_integrals = one_electron
    molecule.two_body_integrals = two_electron

    molecule.nuclear_repulsion = nuc_energy

    H = molecule.get_molecular_hamiltonian()
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(H)
    o = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
    ham = qchem.convert.import_operator(o, wires=wires, tol=(5e-5))
    return ham


def get_active_space_integrals(one_body_integrals, two_body_integrals, occupied_indices=None, active_indices=None):
    """
    Gets integrals in some active space
    """
    # Fix data type for a few edge cases
    occupied_indices = [] if occupied_indices is None else occupied_indices

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        core_constant = core_constant + 2 * one_body_integrals[i][i]
        for j in occupied_indices:
            core_constant = core_constant + (2 * two_body_integrals[i][j][j][i] -
                              two_body_integrals[i][j][i][j])

    # Modified one electron integrals
    one_body_integrals_new = anp.zeros(one_body_integrals.shape)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                c = 2 * two_body_integrals[i][u][v][i] - two_body_integrals[i][u][i][v]
                one_body_integrals_new = one_body_integrals_new + c * build_arr(one_body_integrals.shape, (u, v))

    one_body_integrals_new = one_body_integrals_new + one_body_integrals

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[anp.ix_(active_indices, active_indices)],
            two_body_integrals[anp.ix_(active_indices, active_indices, active_indices, active_indices)])


def hf_energy(num_elec, charge, atomic_orbitals):
    """
    Returns the Hartree-Fock energy
    """
    def energy(atom_R, *args):
        w_fock, fock, h_core, eri_tensor, K, V, S = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        return anp.einsum('pq,qp', fock + h_core, density_matrix(num_elec, w_fock)) + nuclear_energy(charge)(atom_R)
    return energy