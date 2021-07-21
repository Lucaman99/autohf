import numpy as np
from orbitals import AtomicBasisFunction


# basis set

STO3G = {'H':
         {'orbitals' : ['S'],
          'exp'  : [[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]]},
         
         'He':
         {'orbitals' : ['S'],
          'exp'  : [[0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]]},
         
          'Li': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.1611957475E+02, 0.2936200663E+01, 0.7946504870E+00],
                    [0.6362897469E+00, 0.1478600533E+00, 0.4808867840E-01]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
          'Be': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.3016787069E+02, 0.5495115306E+01, 0.1487192653E+01],
                    [0.1314833110E+01, 0.3055389383E+00, 0.9937074560E-01]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
          'B': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.4879111318E+02, 0.8887362172E+01, 0.2405267040E+01],
                    [0.2236956142E+01, 0.5198204999E+00, 0.1690617600E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},

          'C': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.7161683735E+02, 0.1304509632E+02, 0.3530512160E+01],
                    [0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
         
          'N': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.9910616896E+02, 0.1805231239E+02, 0.4885660238E+01],
                    [0.3780455879E+01, 0.8784966449E+00, 0.2857143744E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
         
          'O': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.1307093214E+03, 0.2380886605E+02, 0.6443608313E+01],
                    [0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
         
          'F': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.1666791340E+03, 0.3036081233E+02, 0.8216820672E+01],
                    [0.6464803249E+01, 0.1502281245E+01, 0.4885884864E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]},
         
          'Ne': 
         {'orbitals' : ['S', 'SP'],
          'exp'  : [[0.2070156070E+03, 0.3770815124E+02, 0.1020529731E+02],
                    [0.8246315120E+01, 0.1916266291E+01, 0.6232292721E+00]],
          'coef' : [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                    [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
                    [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]]}
}

basis_sets = {'sto-3g': STO3G}

atomic_numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
                  'N': 7, 'O': 8, 'F': 9, 'Ne': 10}


s = [(0, 0, 0)]
p = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

def get_basis(name, atom):
    
    basis = basis_sets[name][atom]
    params = []
    for i, j in enumerate(basis['orbitals']):
        if j == 'S':
            params.append((s[0], basis['exp'][i], basis['coef'][i]))
        elif j == 'SP':
            for term in j:
                if term == 'S':
                    params.append((s[0], basis['exp'][i], basis['coef'][i]))
                if term == 'P':
                     for l in p:
                        params.append((l, basis['exp'][i], basis['coef'][i+1]))
    return params

def basis_functions(name, symbols):
    """
    """
    basis_set = []
    for s in symbols:
        basis_set += get_basis(name, s)
    return tuple(basis_set)


def generate_basis(l, alpha = None, c = None, r = None):
    """ 
    """

    basis_set = [AtomicBasisFunction(l[i], r[i], c[i], alpha[i]) for i in range(len(l))]

    return basis_set


def electron_number(symbols):
    """
    """
    ne = [atomic_numbers[s] for s in symbols]
    return ne, sum(np.array(ne))

class Molecule:
    
    def __init__(self,
        symbols,
        coordinates,
        charge = 0,
        mult = 1,
        basis_name='sto-3g',
        params = []):
        
        self.symbols = symbols
        self.coordinates = coordinates
        self.charge = charge
        self.mult = mult        
        self.basis_name = basis_name
        
        self.basis_data = basis_functions(self.basis_name, self.symbols)
        
        self.l = [i[0] for i in self.basis_data]
        
        if 'alpha' not in params:
            self.alpha = [i[1] for i in self.basis_data]
        else:
            self.alpha = [None] * len(self.l)

        if 'c' not in params:
            self.c = [i[2] for i in self.basis_data]
        else:
            self.c = [None] * len(self.l)

        if 'r' not in params:
            self.r = [i for i in self.coordinates]
        else:
            self.r = [None] * len(self.l)

        self.n_orbitals = len(self.l)       
        self.nuclear_charges, self.n_electrons = electron_number(symbols)

        
        self.basis_set = generate_basis(l = self.l, alpha = self.alpha, c = self.c, r = self.r)

########## post proc

from orbitals import *
from hartreefock import *
from hamiltonian import *

def energy(ne, nc, R, hf_data):
    
    v, w, F, h_core, eri_tensor = hf_data
    
    return np.einsum('pq,qp', F + h_core, density_matrix(ne, w)) + nuclear_energy(nc)(R)