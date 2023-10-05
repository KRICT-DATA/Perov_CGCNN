import os
from itertools import combinations
A_elems = ["Cs", "K", "Rb"]
B_elems = ['Cd', 'Ge', 'Hg', 'Pb', 'Sn', 'Zn']
X_elems = ["Br", "Cl", "I"]
i = 1
for A in A_elems:
    for X in X_elems:
        for B in B_elems:
            os.system(f"python ../random_searching_restart.py ../0.cif {A} {X} {B} {A}_{B}_{X}.pkl 0 > {A}_{B}_{X}.log")

