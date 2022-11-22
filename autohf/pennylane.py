"""AutoHF + PennyLane

Note that the input "molecules" to each of the below functions must be a class of the form

class MoleculeName:

    # Required
    basis_name = "sto-3g"
    symbols = ["H", "H"]
    active_electrons = 2
    active_orbitals = 2
    charge = 0

    # Optional
    hf_geometry = HARTREE-FOCK GEOMETRY HERE
"""
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
    num = 0

    for b in basis_params:
        t = []
        for func in b:
            L, exp, coeff = func
            t.append(AtomicBasisFunction(L, C=anp.array(coeff), A=anp.array(exp)))
            num += 1
        hf_b.append(t)
    return hf_b, num


def charge_structure(molecule):
    """
    Computes the charge structure of a molecule
    """
    num_elecs, charges = 0, []
    symbols = molecule.symbols

    for s in symbols:
        c = qml.qchem.basis_data.atomic_numbers[s]
        charges.append(c)
        num_elecs += c
    num_elecs -= molecule.charge
    return num_elecs, charges

def V(molecule, wires):
    """
    One-body potential integrals
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    hf_b, num = generate_basis_set(molecule)

    def H_fn(R):
        Ri = R.reshape((len(charges), 3))

        arguments = []
        new_b_set = []
        for i, b in enumerate(hf_b):
            arguments.extend([[Ri[i]]] * len(b))
            new_b_set.extend(b)

        integrals = one_body_potential_integrals(num_elecs, charges, new_b_set)(list(Ri), *arguments)
        return integrals
    return H_fn 

def H(molecule, wires):
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
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H_fn(R):
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
        return build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
    return H_fn


def dH(molecule, wires):
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
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def dH_fn(R, vec):
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
        return build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
    return dH_fn


def ddH(molecule, wires):
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
    hf_b, num = generate_basis_set(molecule) 
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def ddH_fn(R, vec1, vec2):
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
                                         multiplicity=molecule.multiplicity, charge=molecule.charge)
    return ddH_fn


def generate_dd_hamiltonian(molecule, wires, bar=True):
    """
    Generates first Hamiltonian derivatives
    """
    def H(R):
        ddh = ddH(molecule, wires)

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
        dh = dH(molecule, wires)

        H1 = []
        bar_range = tqdm(range(len(R))) if bar else range(len(R))
        for j in bar_range:
            vec = np.array([1.0 if j == k else 0.0 for k in range(len(R))])
            H1.append(dh(R, vec))
        return H1
    return H 


def sparse_H(molecule, wires, guess=None):
    """Generates a sparse Hamiltonian"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals = electron_integrals_flat(num_elecs, charges, new_b_set, guess=guess, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = core_ad + nuc_fn
        return qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec, const=nuc_energy), wires=wires)
    return H


def sparse_H_mat(molecule, guess=None):
    """Generates a sparse Hamiltonian"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals = electron_integrals_flat(num_elecs, charges, new_b_set, guess=guess, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = core_ad + nuc_fn
        return bv.sparse_H(one_elec, two_elec, const=nuc_energy)
    return H


def sparse_H_iter(molecule, wires):
    """Generates a sparse Hamiltonian"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R, initial=None):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals, guess = electron_integrals_flat_new(num_elecs, charges, new_b_set, initial=initial, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = core_ad + nuc_fn
        return qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec, const=nuc_energy), wires=wires), guess
    return H

def w_coeffs(molecule, guess=None):
    num_elecs, charges = charge_structure(molecule)
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        w = hf_coeffs(num_elecs, charges, new_b_set, guess=guess, occupied=core, active=active)(
            *transform(R))

        return w
    return H  

def sparse_H_w(molecule, wires, w):
    """Generates a sparse Hamiltonian"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            re_hf = re_fn(molecule.hf_geometry)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals = electron_integrals_flat_w(num_elecs, charges, new_b_set, w, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = core_ad + nuc_fn
        return qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec, const=nuc_energy), wires=wires)
    return H


def sparse_dH(molecule, wires):
    """
    Generates the sparse representation of a hamiltonian derivative with respect to
    a Hamiltonian using the BigVQE and AutoHF libraries.

    Args
        molecule: chemistry.Molecule object
    Returns
        scipy.coo_matrix
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def dH(R, vec):
        re_fn = lambda r : r.reshape((len(charges), 3))

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
        return qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec, const=nuc_energy), wires=wires)
    return dH


def hf_energy_molecule(molecule):
    """
    Finds the Hartree-Fock energy of a molecule
    """
    # Basic info
    num_elecs, charges = charge_structure(molecule)
    hf_b, num = generate_basis_set(molecule)
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
        
    E = lambda R : energy_fn(*transform(R))
    return E


def universal(molecule, wires):
    """Universal functional"""
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b, num = generate_basis_set(molecule)
    core, active = qchem.active_space(num_elecs, num, active_electrons=molecule.active_electrons, active_orbitals=molecule.active_orbitals)  # Prepares active space

    def H(R):
        re_fn = lambda r: r.reshape((len(charges), 3))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        integrals = universal_electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active)(
            *transform(R))

        n = len(active)
        num = (n ** 2)
        core_ad, one_elec, one_elec_v, one_elec_s, two_elec = integrals[0], integrals[1:num + 1].reshape((n, n)), integrals[num + 1:2*num + 1].reshape((n, n)), integrals[2*num + 1:3*num + 1].reshape((n, n)), integrals[3*num + 1:].reshape(
            (n, n, n, n))

        nuc_fn = nuclear_energy(charges)(re_fn(R))
        nuc_energy = nuc_fn + core_ad
        
        uni = qml.SparseHamiltonian(bv.sparse_H(one_elec, two_elec), wires=wires)

        return uni, one_elec_v, one_elec_s, nuc_energy
    return H 


def hf_energy_gradient(molecule):
    """
    Computes the gradient of the Hartree-Fock energy with respect to nuclear coordinates
    """
    fn = hf_energy_molecule(molecule)
    gradient = lambda R, vec : make_jvp(fn)(R)(vec)[1] 
    return gradient


def hf_geometry(molecule, initial_geo, steps, epsilon=0.05, bar=True):
    """
    Finds the Hartree-Fock geometry of a molecule
    """
    geo = initial_geo
    bar_range = tqdm(range(steps)) if bar else range(steps) 
    grad = hf_energy_gradient(molecule)

    for s in bar_range:
        grad_vector = np.array([grad(geo, np.array([1.0 if i == j else 0.0 for i in range(len(geo))])) for j in range(len(geo))])
        geo = geo - epsilon * grad_vector
    return geo
