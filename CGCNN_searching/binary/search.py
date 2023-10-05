import os
from itertools import combinations
A_elems = ["Cs", "K", "Rb"]
B_elems = ['Cd', 'Ge', 'Hg', 'Pb', 'Sn', 'Zn']
X_elems = ["Br", "Cl", "I"]
binarys = list(combinations(B_elems, 2))
i = 1
for A in A_elems:
    for X in X_elems:
        for binary in binarys:
            file = f"job{i}.sh"
            os.system(f"cp ../job_submit_example.sh {file}")
            B1=binary[0]
            B2=binary[1]
            os.system(f"echo 'python ../random_searching_restart.py ../0.cif {A} {X} {B1},{B2} {A}_{B1}{B2}_{X}.pkl 0 > {A}_{B1}{B2}_{X}.log' >> {file}")
            os.system(f"sed -i '3s/.*/#SBATCH --job-name={A}_{B1}{B2}_{X}/g' {file}")
            os.system(f"sbatch {file}")
            os.system("sleep 0.1")
            i += 1

