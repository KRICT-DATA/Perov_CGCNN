import pickle, os, re
import numpy as np
from itertools import combinations
from ase.io import write
import pandas as pd
script_dir = os.getcwd()
A_elems = ["Cs", "K", "Rb"]
B_elems = ['Cd', 'Ge', 'Hg', 'Pb', 'Sn', 'Zn']
X_elems = ["Br", "Cl", "I"]
n = {'Cs': 1, 'Rb': 1, 'K': 1, 'Cd': 2, 'Ge': 2, 'Hg': 2, "Pb": 2, 'Sn': 2, 'Zn': 2, 'Br': -1, 'Cl': -1, 'I': -1}
r = {'Cs': 1.88, 'Rb': 1.72, 'K': 1.64, 'Cd': 0.95, 'Ge': 0.73, 'Hg': 1.02, "Pb": 1.19, 'Sn': 1.15, 'Zn': 0.74,
          'Br': 1.96, 'Cl': 1.81, 'I': 2.2}
binarys = list(combinations(B_elems, 2))
ternarys = list(combinations(B_elems, 3))
quaternarys = list(combinations(B_elems, 4))
k_B = 8.61736292496E-5
T = 298
clf_name = 'bandtype_model_best.pth.tar'
reg_name = 'bandgap_model_best.pth.tar'

def calc_entropy(ratios):
    entropy = 0
    for ratio in ratios:
        if ratio == 0.0:
            entropy += 0
        else:
            entropy += ratio*np.log(ratio)
    return entropy

os.system("mkdir band")
with open(f"./band/id_prop.csv", 'w'):
    pass
df = pd.DataFrame(columns=["Elements","deltaG","tau","Band type","Band gap"])
for A in A_elems:
    for X in X_elems:
        for binary in binarys:
            B1=binary[0]
            B2=binary[1]
            with open(f"./binary/pickle_files/{A}_{B1}{B2}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            elems, deltaGs, taus = [], [], []
            bandtypes, bandgaps = [0 for i in range(len(Results))], [0 for i in range(len(Results))]
            num = 0
            for i in range(1, 16):
                ratio1 = float(i/16)
                ratio2 = 1-ratio1
                rB = float(i/16) * r[B1] + float((16-i)/16) * r[B2]
                tau = r[X] / (rB) - n[A] * (n[A] - r[A] / (rB * np.log(r[A] / rB)))
                deltaH = Results[f'{i}_{16-i}']['Prediction'][0]
                deltaS = 1000*k_B*T*calc_entropy([ratio1, ratio2])
                deltaG = deltaH + deltaS
                elems.append(f"{A}.{B1}.{i}.{B2}.{16-i}.{X}"), deltaGs.append(deltaG), taus.append(tau)
                atoms = Results[f"{i}_{16-i}"]['MinConfig'][0]
                write(f"./band/{num}.cif", atoms)
                with open(f"./band/id_prop.csv", 'a') as f:
                    if num % 2 == 0:
                        f.write(f"{num},0\n")
                    else:
                        f.write(f"{num},1\n")
                num += 1
            os.system("cp ./atom_init.json ./band/")
            os.system(f"python ./predict.py {clf_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                if res.iloc[j, 2] <= 0.5:
                    bandtypes[res.iloc[j, 0]] = "Non-direct"
                else:
                    bandtypes[res.iloc[j, 0]] = "Direct"
            os.system(f"python ./predict.py {reg_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                bandgaps[res.iloc[j, 0]] = float(res.iloc[j, 2])
            os.system("rm ./band/*")
            df_sub = pd.DataFrame({"Elements": elems, "deltaG": deltaGs, "tau": taus, "Band type": bandtypes, "Band gap": bandgaps})
            df = pd.concat([df, df_sub], axis=0, ignore_index=True)

        for ternary in ternarys:
            B1,B2,B3 = ternary[0],ternary[1], ternary[2]
            with open(f"./ternary/pickle_files/{A}_{B1}{B2}{B3}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            elems, deltaGs, taus = [], [], []
            bandtypes, bandgaps = [0 for i in range(len(Results))], [0 for i in range(len(Results))]
            num = 0
            for i in range(1, 16):
                for j in range(1, 16-i):
                    k = 16 - i - j
                    r1, r2, r3 = float(i/16), float(j/16), float(k/16)
                    rB = r1*r[B1] + r2*r[B2] + r3*r[B3]
                    tau = r[X] / (rB) - n[A] * (n[A] - r[A] / (rB * np.log(r[A] / rB)))
                    deltaH = Results[f'{i}_{j}_{k}']['Prediction'][0]
                    deltaS = 1000*k_B*T*calc_entropy([r1, r2, r3])
                    deltaG = deltaH + deltaS
                    elems.append(f"{A}.{B1}.{i}.{B2}.{j}.{B3}.{k}.{X}"), deltaGs.append(deltaG), taus.append(tau)
                    atoms = Results[f"{i}_{j}_{k}"]['MinConfig'][0]
                    write(f"./band/{num}.cif", atoms)
                    with open(f"./band/id_prop.csv", 'a') as f:
                        if num % 2 == 0:
                            f.write(f"{num},0\n")
                        else:
                            f.write(f"{num},1\n")
                    num += 1
            os.system("cp ./atom_init.json ./band/")
            os.system(f"python ./predict.py {clf_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                if res.iloc[j, 2] <= 0.5:
                    bandtypes[res.iloc[j, 0]] = "Non-direct"
                else:
                    bandtypes[res.iloc[j, 0]] = "Direct"
            os.system(f"python ./predict.py {reg_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                bandgaps[res.iloc[j, 0]] = float(res.iloc[j, 2])
            os.system("rm ./band/*")
            df_sub = pd.DataFrame({"Elements": elems, "deltaG": deltaGs, "tau": taus, "Band type": bandtypes, "Band gap": bandgaps})
            df = pd.concat([df, df_sub], axis=0, ignore_index=True)

        for quaternary in quaternarys:
            B1,B2,B3,B4 = quaternary[0], quaternary[1], quaternary[2], quaternary[3]
            if A == 'Cs':
                with open(f"./quaternary/pickle_files/{A}_{B1}{B2}{B3}{B4}_{X}.pkl", 'rb') as f:
                    Results = pickle.load(f)
            else:
                continue
            elems, deltaGs, taus = [], [], []
            bandtypes, bandgaps = [0 for i in range(455)], [0 for i in range(455)]
            num = 0
            flag = False
            for i in range(1, 16):
                for j in range(1, 16-i):
                    for k in range(1, 16-i-j):
                        l = 16 - i - j - k
                        r1, r2, r3, r4 = float(i/16), float(j/16), float(k/16), float(l/16)
                        deltaS = 1000*k_B*T*calc_entropy([r1, r2, r3, r4])
                        rB = r1*r[B1] + r2*r[B2] + r3*r[B3] + r4*r[B4]
                        tau = r[X] / (rB) - n[A] * (n[A] - r[A] / (rB * np.log(r[A] / rB)))
                        if 'Prediction' in Results[f'{i}_{j}_{k}_{l}']:
                            deltaH = Results[f'{i}_{j}_{k}_{l}']['Prediction'][0]
                        else:
                            deltaH = Results[f'{i}_{j}_{k}_{l}']['Min']
                        deltaG = deltaH + deltaS
                        elems.append(f"{A}.{B1}.{i}.{B2}.{j}.{B3}.{k}.{B4}.{l}.{X}"), deltaGs.append(deltaG), taus.append(tau)
                        if 'MinConfig' in Results[f'{i}_{j}_{k}_{l}']:
                            atoms = Results[f"{i}_{j}_{k}_{l}"]['MinConfig'][0]
                        else:
                            atoms = Results[f'{i}_{j}_{k}_{l}']['MinConf'][0]
                        write(f"./band/{num}.cif", atoms)
                        with open(f"./band/id_prop.csv", 'a') as f:
                            if num % 2 == 0:
                                f.write(f"{num},0\n")
                            else:
                                f.write(f"{num},1\n")
                        num += 1
            os.system("cp ./atom_init.json ./band/")
            os.system(f"python ./predict.py {clf_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                if res.iloc[j, 2] <= 0.5:
                    bandtypes[res.iloc[j, 0]] = "Non-direct"
                else:
                    bandtypes[res.iloc[j, 0]] = "Direct"
            os.system(f"python ./predict.py {reg_name} ./band/")
            res = pd.read_csv('test_results.csv', header=None)
            for j in range(len(res)):
                bandgaps[res.iloc[j, 0]] = float(res.iloc[j, 2])
            os.system("rm ./band/*")
            df_sub = pd.DataFrame({"Elements": elems, "deltaG": deltaGs, "tau": taus, "Band type": bandtypes, "Band gap": bandgaps})
            df = pd.concat([df, df_sub], axis=0, ignore_index=True)
os.system("rm -r band")
df.to_csv('All_data.csv', index=False)
