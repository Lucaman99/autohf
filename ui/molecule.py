import numpy as np
from basis_data import atomic_numbers
from basis_set import get_basis, basis_functions

class Molecule:
    
    def __init__(self,
        symbols,
        coordinates,
        charge = 0,
        mult = 1,
        basis_name='sto-3g',
        a = False,
        c = False,
        R_diff = False):
        
        self.symbols = symbols
        self.coordinates = coordinates
        self.charge = charge
        self.mult = mult        
        self.basis_name = basis_name
        
        self.n_basis, self.basis_data = get_basis(self.basis_name, self.symbols)
        
        self.l = [i[0] for i in self.basis_data]

        self.a_init = [i[1] for i in self.basis_data]
        
        self.c_init = [i[2] for i in self.basis_data]
        
        r_atom = [i for i in self.coordinates]
        self.r_init = sum([[r_atom[i]] * self.n_basis[i] for i in range(len(self.n_basis))], [])

        
        if a:
            exp = [None] * len(self.l)
        else:
            exp = self.a_init
        if c:
            coef = [None] * len(self.l)
        else:
            coef = self.c_init
        if R_diff:
            coor = [None] * len(self.l)
        else:
            coor = self.r_init
        
        self.basis_set = basis_functions(self.l, exp, coef, coor)

        self.n_orbitals = len(self.l)       
        self.nuclear_charges, self.n_electrons = electron_number(symbols)
        


def electron_number(symbols):
    """
    """
    ne = [atomic_numbers[s] for s in symbols]
    return ne, sum(np.array(ne))


# def generate_params(R_params, params_r=None, params_c=None, params_a=None):
    
#     print()
    
#     list_par = [i for i in [params_r, params_c, params_a] if i is not None]
    
#     params = [[0.0]] * 6

#     for i in range(len(params)):
#         params[i] = [j[i][0] for j in list_par]
        
#     return R_params, params

def generate_params(mol, R_params=None, params_r=None, params_c=None, params_a=None):
    
    if R_params is None:
        R_params = [np.array(i) for i in mol.coordinates]
    
    list_par = [i for i in [params_r, params_c, params_a] if i is not None]
    
    params = [[0.0]] * sum(mol.n_basis)

    for i in range(len(params)):
        params[i] = [j[i][0] for j in list_par]
        
    return R_params, params