import numpy as np
import pickle
import time
import sys

from itertools import combinations
from itertools import combinations_with_replacement

import torch
import numpy as np
from tqdm import tqdm

from ase.io import read
from ase.data import atomic_numbers

from pymatgen.core.structure import Structure
from pymatgen.io.ase import *

from model import CrystalGraphConvNet
import my_utils

from multiprocessing import Pool
                                                                                                                                      
### Global Variables ###
CIFNAME = sys.argv[1]
atoms = read(CIFNAME)
Natom = atoms.get_global_number_of_atoms()
NumA = int(Natom/5)
NumB = int(Natom/5)
NumC = int(NumA*3)
E_A = sys.argv[2]
E_C = sys.argv[3]
E_Bs = sys.argv[4].split(',')
SAVENAME = sys.argv[5]
Va = my_utils.GetEmb(E_A,NumA)
Vc = my_utils.GetEmb(E_C,NumC)
EdgeFea,EdgeIdx = my_utils.GetEdgeFea(cif_name=CIFNAME,radius=8,max_num_nbr=12,dmin=0,step=0.2)
########################

### Load ChkFile ###
restart = bool(float(sys.argv[6]))
if restart:
        chkptfile = sys.argv[7]
###################

### Model Load #######################################################
chkpt_name = 'deltaH_decomp_model_best.pth.tar'
chkpt = torch.load(chkpt_name)
norm = chkpt['normalizer']
model = CrystalGraphConvNet(orig_atom_fea_len = Va.shape[1],
                                                        nbr_fea_len = EdgeFea.shape[-1],
                                                        atom_fea_len = chkpt['args']['atom_fea_len'],
                                                        n_conv = chkpt['args']['n_conv'],
                                                        h_fea_len = chkpt['args']['h_fea_len'],
                                                        n_h = chkpt['args']['n_h'])
model.load_state_dict(chkpt['state_dict'])
#######################################################################

def LoadChkFile(chkptfile):
        lines = open(chkptfile).readlines()
        tmp = []
        for ll in lines:
                if 'Start for' in ll:
                        tmp.append(ll.split()[2])
        keys_done = tmp[:-1]
        return keys_done

def GetCase_H():
        site_base = [i+1 for i in range(len(E_Bs))]
        Ns = []

        for cc_ in tqdm(combinations_with_replacement(site_base,NumB)):
                cc = list(cc_)
                if len(list(set(cc))) < len(site_base):
                        continue
                ns = [cc.count(i) for i in site_base]
                Ns.append(ns)
        return Ns
def GetAtoms(cs):
        atoms = read(CIFNAME)
        Bsite = [0]*NumB
        for k in cs:
                for i in cs[k]:
                        Bsite[i] = atomic_numbers[k]

        Asite = [atomic_numbers[E_A]]*NumA
        Csite = [atomic_numbers[E_C]]*NumC
        AllSites = Asite+Bsite+Csite
        atoms.set_atomic_numbers(AllSites)
        return atoms

def GetCase_C4(ns):
        Confs = []
        Nb = np.sum(ns)
        base = [i for i in range(NumB)]
        for s1_ in tqdm(combinations(base,ns[0])):
                s1 = list(s1_)
                base2 = [i for i in base if not i in s1]

                for s2_ in combinations(base2,ns[1]):
                        s2 = list(s2_)
                        base3 = [i for i in base2 if not i in s2]

                        for s3_ in combinations(base3,ns[2]):
                                s3 = list(s3_)
                                s4 = [i for i in base3 if not i in s3]

                                s = {k:v for k,v in zip(E_Bs,[s1,s2,s3,s4])}
                                Confs.append(s)
        return Confs

def GetCase_C3(ns):
        Confs = []
        Nb = np.sum(ns)
        base = [i for i in range(NumB)]
        for s1_ in tqdm(combinations(base,ns[0])):
                s1 = list(s1_)
                base2 = [i for i in base if not i in s1]

                for s2_ in combinations(base2,ns[1]):
                        s2 = list(s2_)
                        s3 = [i for i in base2 if not i in s2]

                        s = {k:v for k,v in zip(E_Bs,[s1,s2,s3])}
                        Confs.append(s)
        return Confs

def GetCase_C2(ns):
        Confs = []
        base = [i for i in range(NumB)]
        for s1_ in tqdm(combinations(base,ns[0])):
                s1 = list(s1_)
                s2 = [i for i in base if not i in s1]

                s = {k:v for k,v in zip(E_Bs,[s1,s2])}
                Confs.append(s)
        return Confs

def GetCase_C1(ns):
        Confs = []
        base = [i for i in range(NumB)]
        for s1_ in tqdm(combinations(base,ns[0])):
                s1 = list(s1_)
                s2 = [i for i in base if not i in s1]

                s = {k:v for k,v in zip(E_Bs,[s1,s2])}
                Confs.append(s)
        return Confs


def myfunc(cs):
        Embs = my_utils.embs
        F = Embs.shape[1]
        Vb = np.zeros((NumB,F))
        for k in cs:
                id_k = cs[k]
                Vb[id_k] = Embs[atomic_numbers[k]]

        node = np.vstack([Va,Vb,Vc])
        node_fea = torch.Tensor(np.vstack([Va,Vb,Vc]))
        nbr_fea = torch.Tensor(EdgeFea)
        nbr_idx = torch.LongTensor(EdgeIdx)
        crys_idx = [np.arange(Natom)]

        with torch.no_grad():
                inp_var = (node_fea,nbr_fea,nbr_idx,crys_idx)
                model.eval()
                out = model(*inp_var)
                out_den = norm['mean']+out*norm['std']
                out_den = out_den.cpu().detach().numpy()
        return cs,out_den

print('Summary of calculations')
print('Site name (A,B,C)...',E_A,','.join(E_Bs),E_C)
print('Base cif file name...',CIFNAME)
print('Save file name...',SAVENAME)

_T1 = time.time()

t1 = time.time()
Hs = GetCase_H()
t2 = time.time()

print('Fin; combination with repetition...time:',t2-t1)

if restart:
        keys_done = LoadChkFile(chkptfile)
else:
        keys_done = []

Results = {}
for CaseID,hs in enumerate(Hs):
        key = '_'.join([str(h) for h in hs])
        if key in keys_done:
                print('Start for ',key,'['+str(CaseID+1)+'/'+str(len(Hs))+']')
                continue

        print('Start for ',key,'['+str(CaseID+1)+'/'+str(len(Hs))+']')

        t1 = time.time()
        if len(E_Bs) == 2:
                cs = GetCase_C2(hs)
        elif len(E_Bs) == 3:
                cs = GetCase_C3(hs)
        elif len(E_Bs) == 4:
                cs = GetCase_C4(hs)
        elif len(E_Bs) == 1:
                cs = GetCase_C1(hs)
        t2 = time.time()

        print('Fin;generate all combinations',len(cs),'...time:',t2-t1)

        t1 = time.time()
        with Pool(56) as pool:
                res = pool.map(myfunc,cs)
        t2 = time.time()
        print('Fin; energy prediction...time:',t2-t1)

        all_cs = [rr[0] for rr in res]
        egy = np.vstack([rr[1] for rr in res]).flatten()
        ids, = np.where(egy==np.min(egy))

        vals = [np.min(egy),np.max(egy),np.mean(egy),np.std(egy)]
        min_conf = [GetAtoms(all_cs[i]) for i in ids]
        print('Egy stat',vals)

        Results[key] = {'Prediction':vals,'MinConfig':min_conf}
        '''
        if len(E_Bs) == 4:
                temp = {}
                temp[key] = {'Prediction':vals,'MinConfig':min_conf}
                with open(f"./pkl_band/{E_A}_{E_Bs[0]}{hs[0]}{E_Bs[1]}{hs[1]}{E_Bs[2]}{hs[2]}{E_Bs[3]}{hs[3]}_{E_C}.pkl", 'wb') as f :
                        pickle.dump(temp, f)
        '''

print('Saving....')
with open(SAVENAME,'wb') as f:
        pickle.dump(Results,f)
print('Saved to...',SAVENAME)
_T2 = time.time()
print('Total elapsed time...',_T2-_T1)
