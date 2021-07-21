"""
Post-processing of Hartree-Fock results and generation of Hamiltonians
"""
import autograd.numpy as anp
from hartreefock import *
from utils import cartesian_prod, build_arr
import openfermion
from pennylane import qchem


def one_electron_integral(num_elec, charge, atomic_orbitals):
    """Returns the one electron integral matrix"""
    def one_elec(atom_R, *args):
        v_fock, w_fock, fock, h_core, eri_tensor = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        t = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        return t
    return one_elec


def two_electron_integral(num_elec, charge, atomic_orbitals):
    """Returns the two electron integral matrix"""
    def two_elec(atom_R, *args):
        v_fock, w_fock, fock, h_core, eri_tensor = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        t = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        return t
    return two_elec


def one_electron_integral_flat(num_elec, charge, atomic_orbitals):
    """One electron integrals flat"""
    return lambda atom_R, *args : one_electron_integral(num_elec, charge, atomic_orbitals)(atom_R, *args).flatten()


def two_electron_integral_flat(num_elec, charge, atomic_orbitals):
    """Two electron integrals flat"""
    return lambda atom_R, *args : two_electron_integral(num_elec, charge, atomic_orbitals)(atom_R, *args).flatten()


def one_electron_integral_entry(num_elec, charge, atomic_orbitals, idx):
    """Returns the one electron integrals entry"""
    return lambda atom_R, *args : one_electron_integral(num_elec, charge, atomic_orbitals)[idx[0]][idx[1]]


def two_electron_integral_entry(num_elec, charge, atomic_orbitals, idx):
    """Returns two electron integrals entry"""
    return lambda atom_R, *args: two_electron_integral(num_elec, charge, atomic_orbitals)[idx[0]][idx[1]][idx[2]][idx[3]]


def electron_integrals(num_elec, charge, atomic_orbitals):
    """Returns the one and two electron integrals"""
    def I(atom_R, *args):
        v_fock, w_fock, fock, h_core, eri_tensor = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        return one, two
    return I


def electron_integrals_flat(num_elec, charge, atomic_orbitals, occupied=None, active=None):
    """Returns the one and two electron integrals flattened into a 1D array"""
    def I(atom_R, *args):
        v_fock, w_fock, fock, h_core, eri_tensor = hartree_fock(num_elec, charge, atomic_orbitals)(atom_R, *args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)

        core, one_elec, two_elec = get_active_space_integrals(one, two, occupied_indices=occupied, active_indices=active)
        return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten()))
    return I


def distance(pair):
    return anp.sqrt(((pair[0] - pair[1]) ** 2).sum())


def nuclear_energy(charges):
    """
    Generates the repulsion between nuclei of the atoms in a molecule
    """
    def repulsion(R):
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
    ham = qchem.convert_observable(o, wires=wires, tol=(5e-5))
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

