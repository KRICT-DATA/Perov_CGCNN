import numpy as np
import json

from ase.io import read
from ase.data import atomic_numbers

from pymatgen.core.structure import Structure
from pymatgen.io.ase import *

atom_init = json.load(open('atom_init.json'))
N = len(atom_init)+1
F = len(atom_init['1'])
embs = np.zeros((N,F))

for i in range(1,101):
	embs[i] = np.array(atom_init[str(i)])

def GetEmb(Ele,Num):
	Ve = embs[atomic_numbers[Ele]].reshape(1,-1).repeat(Num,0)
	return Ve

def GetGauss(dmin,dmax,step,distances,var=None):
	if var is None:
		var = step
	
	filter = np.arange(dmin,dmax+step,step)
	fea = np.exp(-(distances[..., np.newaxis] - filter)**2 /var**2)
	return fea

def GetEdgeFea(cif_name,radius,max_num_nbr,dmin,step):
	atoms = read(cif_name)
	crystal = AseAtomsAdaptor.get_structure(atoms)
	all_nbrs = crystal.get_all_neighbors(radius,include_index=True)
	all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
	Nbr_Idx = np.array([list(map(lambda x: x[2],nbr[:max_num_nbr])) for nbr in all_nbrs])
	Dij = np.array([list(map(lambda x: x[1], nbr[:max_num_nbr])) for nbr in all_nbrs])

	Nbr_Fea = GetGauss(dmin,radius,step,Dij,var=None)
	return Nbr_Fea,Nbr_Idx

