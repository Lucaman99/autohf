"""AutoHF + PennyLane"""
import pennylane as qml
import numpy as np
import autograd.numpy as anp
import autograd
from autograd.differential_operators import make_jvp
from tqdm.notebook import tqdm
import bigvqe as bv
from pennylane import qchem

from .hamiltonian import *
from .orbitals import *


angs_bohr = 1.8897259885789

def generate_basis_set(molecule):
    """
    Generates a basis set corresponding to a molecule
    """
    basis_name = molecule.basis_name
    structure = molecule.symbols
    basis_params = basis_set_params(basis_name, structure)
    hf_b = []

    for b in basis_params:
        t = []
        for func in b:
            L, exp, coeff = func
            t.append(AtomicBasisFunction(L, C=anp.array(coeff), A=anp.array(exp)))
        hf_b.append(t)
    return hf_b


def charge_structure(molecule):
    """
    Computes the charge structure of a molecule
    """
    num_elecs, charges = 0, []
    symbols = molecule.symbols

    for s in symbols:
        c = qml.hf.basis_data.atomic_numbers[s]
        charges.append(c)
        num_elecs += c
    num_elecs -= molecule.charge
    return num_elecs, charges


def hamiltonian(molecule, wires):
    """
    Computes an electronic Hamiltonian with Hartree-Fock, using the AutoHF library.
    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, len(hf_b), active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        Ri = R.reshape((len(charges), 3))
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        arguments = []
        new_b_set = []
        for i, b in enumerate(hf_b):
            arguments.extend([[Ri[i]]] * len(b))
            new_b_set.extend(b)

        integrals = electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active)(list(Ri), *arguments)

        n = len(active)
        num = (n ** 2) + 1

        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape((n, n, n, n))
        nuc_energy = core_ad + nuclear_energy(charges)(Ri)
        return build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.mult, charge=molecule.charge)
    return H


def d_hamiltonian(molecule, wires):
    """
    Computes an exact derivative of an electronic Hamiltonian with respect to nuclear coordinates using the
    AutoHF library.

    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, len(hf_b), active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def dH(R, vec):
        re_fn = lambda r : r.reshape((len(charges), 3))
        Ri = re_fn(R)
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        fn = lambda r : electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active)(*transform(r))
        integrals = make_jvp(fn)(R)(vec)[1]

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = autograd.grad(lambda r : nuclear_energy(charges)(re_fn(r)))(R)
        nuc_energy = core_ad + np.dot(nuc_fn, vec)
        return build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.mult, charge=molecule.charge)
    return dH


def dd_hamiltonian(molecule, wires):
    """
    Computes an exact second derivative of an electronic Hamiltonian with respect to nuclear coordinates, using the
    AutoHF library.

    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule) 
    core, active = qchem.active_space(num_elecs, len(hf_b), active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def ddH(R, vec1, vec2):
        re_fn = lambda r: r.reshape((len(charges), 3))
        Ri = re_fn(R)
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        fn = lambda r: electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active)(
            *transform(r))
        d_fn = lambda r : make_jvp(fn)(r)(vec1)[1]
        integrals = make_jvp(d_fn)(R)(vec2)[1]

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = autograd.hessian(lambda r: nuclear_energy(charges)(re_fn(r)))(R)
        nuc_energy = core_ad + np.dot(vec1, nuc_fn @ vec2)
        return build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis,
                                         multiplicity=molecule.mult, charge=molecule.charge)
    return ddH


def generate_dd_hamiltonian(molecule, wires, bar=True):
    """
    Generates first Hamiltonian derivatives
    """
    def H(R):
        ddh = dd_hamiltonian(molecule, wires)

        H2 = [[0 for l in range(len(R))] for k in range(len(R))]
        for j in range(len(R)):
            bar_range = tqdm(range(len(R))) if bar else range(len(R))
            for k in bar_range:
                if j <= k:
                    vec1 = np.array([1.0 if j == l else 0.0 for l in range(len(R))])
                    vec2 = np.array([1.0 if k == l else 0.0 for l in range(len(R))])
                    val = ddh(R, vec1, vec2)
                    H2[j][k], H2[k][j] = val, val
        return H2
    return H


def generate_d_hamiltonian(molecule, wires, bar=True):
    """
    Generates second Hamiltonian derivatives
    """
    def H(R):
        dh = d_hamiltonian(molecule, wires)

        H1 = []
        bar_range = tqdm(range(len(R))) if bar else range(len(R))
        for j in bar_range:
            vec = np.array([1.0 if j == k else 0.0 for k in range(len(R))])
            H1.append(dh(R, vec))
        return H1
    return H 


def hamiltonian_sparse(molecule, wires):
    """Generates a sparse Hamiltonian"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, len(hf_b), active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals = electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = core_ad + nuc_fn
        return qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec, const=nuc_energy), wires=wires)
    return H


def hf_energy_molecule(molecule):
    """
    Finds the Hartree-Fock energy of a molecule
    """
    # Basic info
    num_elecs, charges = charge_structure(molecule)
    hf_b = generate_basis_set(molecule)
    new_b_set = sum(hf_b, [])

    # Energy function
    energy_fn = hf_energy(num_elecs, charges, new_b_set)

    # New energy function
    re_fn = lambda r: r.reshape((len(charges), 3))

    def transform(r):
        re = re_fn(r)
        arguments = []
        for i, b in enumerate(hf_b):
            arguments.extend([[re[i]]] * len(b))
        return re, *arguments
        
    E = lambda R : hf_energy(*transform(R))
    return E
