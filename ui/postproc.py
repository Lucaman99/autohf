from hartreefock import density_matrix, hartree_fock
from hamiltonian import nuclear_energy
import autograd.numpy as anp
import pennylane as qml
import numpy as np
import openfermion


def energy(mol):
    """
    """
    n_electrons = mol.n_electrons
    nuclear_charges = mol.nuclear_charges
    coordinates = [np.array(i) for i in mol.coordinates]
    atomic_orbitals = mol.basis_set
    
    def total_energy(*args):
        v, w, F, h_core, eri_tensor = hartree_fock(n_electrons, nuclear_charges, atomic_orbitals)(*args)
        e_elec = anp.einsum('pq,qp', F + h_core, density_matrix(n_electrons, w))
        e_nuc = nuclear_energy(nuclear_charges)(args[0])
        return e_elec , e_nuc
    return total_energy


def energy_hf(mol):
    """
    """
    n_electrons = mol.n_electrons
    nuclear_charges = mol.nuclear_charges
    coordinates = [np.array(i) for i in mol.coordinates]
    atomic_orbitals = mol.basis_set
    
    def total_energy(*args):
        v, w, F, h_core, eri_tensor = hartree_fock(n_electrons, nuclear_charges, atomic_orbitals)(*args)
        e_elec = anp.einsum('pq,qp', F + h_core, density_matrix(n_electrons, w))
        e_nuc = nuclear_energy(nuclear_charges)(args[0])
        return e_elec + e_nuc
    return total_energy


def electron_integrals(mol):
    """Returns the one and two electron integrals"""
    num_elec = mol.n_electrons
    atomic_orbitals = mol.basis_set
    nuclear_charges = mol.nuclear_charges
    
    def integrals(*args):
        v_fock, w_fock, fock, h_core, eri_tensor = hartree_fock(num_elec, nuclear_charges, atomic_orbitals)(*args)
        one = anp.einsum("qr,rs,st->qt", w_fock.T, h_core, w_fock)
        two = anp.swapaxes(anp.einsum("ab,cd,bdeg,ef,gh->acfh", w_fock.T, w_fock.T, eri_tensor, w_fock, w_fock), 1, 3)
        return one, two
    return integrals


def hamiltonian(mol, one_electron, two_electron, nuc_energy):
    
    geometry = [[mol.symbols[i], mol.coordinates[i]] for i in range(len(mol.coordinates))] 
    wires = [i for i in range(mol.n_orbitals * 2)]
    basis = mol.basis_name
    multiplicity = mol.mult
    charge = mol.charge
    
    # Prepares an OpenFermion molecule with input integrals
    molecule = openfermion.MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
    molecule.one_body_integrals = one_electron
    molecule.two_body_integrals = two_electron
    
    # Sets the nuclear repulsion energy with input
    molecule.nuclear_repulsion = nuc_energy
    # Builds the Hamiltonian
    H = molecule.get_molecular_hamiltonian()
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(H)
    o = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
    ham = qml.qchem.convert_observable(o, wires=wires)
    return ham