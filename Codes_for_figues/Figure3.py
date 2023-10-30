import pandas as pd
import numpy as np
from collections import Counter
gas_constant: float = 8.61736292496E-5
Temperature = 298
def calc_entropy(ratios):
    entropy = 0
    for ratio in ratios:
        if ratio == 0.0:
            entropy += 0
        else:
            entropy += ratio*np.log(ratio)
    return entropy
df = pd.read_excel('../Data/Training data for CGCNN.xlsx')
entropy = []
df = df.rename(columns={'Elements (A.B1.n1.B2.n2.B3.n3.B4.n4.X)': 'Elements'})
for i in range(len(df)):
    A = df.iloc[i]['Elements'].split('.')[0]
    B1 = df.iloc[i]['Elements'].split('.')[1]
    B2 = df.iloc[i]['Elements'].split('.')[2]
    B3 = df.iloc[i]['Elements'].split('.')[3]
    B4 = df.iloc[i]['Elements'].split('.')[4]
    X = df.iloc[i]['Elements'].split('.')[5]
    counter = dict(Counter([B1, B2, B3, B4]))
    B_ = list({B1, B2, B3, B4})
    ratios = []
    for B in B_:
        ratios.append(counter[B]*0.25)
    entropy.append(1000*gas_constant*Temperature*calc_entropy(ratios))
df['TdeltaS (meV/atom)'] = entropy
df['deltaG (meV/atom)'] = df['deltaH_decomposition (meV/atom)'] + entropy
import matplotlib.pyplot as plt
fontdict = {'font': 'Times New Roman', 'size': 36, 'fontweight': 'bold'}
thickness = 2

idx = df[(df['tau']>4.3) & (df['deltaG (meV/atom)']<-75)].index
print(df.loc[idx]['Elements'])
idx = df[(df['tau']<4.18) & (df['deltaG (meV/atom)']>0)].index
print(df.loc[idx]['Elements'])

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
for i in range(2):
    axs[i].spines['left'].set_linewidth(thickness)
    axs[i].spines['right'].set_linewidth(thickness)
    axs[i].spines['top'].set_linewidth(thickness)
    axs[i].spines['bottom'].set_linewidth(thickness)
    axs[i].tick_params(length=5, width=thickness)
axs[0] = plt.subplot(1, 2, 1)
plt.scatter(x=df['tau'], y=df['deltaG (meV/atom)'], c='black', alpha=0.5)
plt.xlabel(r"$\mathregular{\tau}$", **fontdict)
plt.xticks(**fontdict)
plt.ylabel(r"$\mathregular {\Delta H_{decomp}-T\Delta S_{mix}}$"' (meV/atom)', **fontdict)
print(df['tau'].corr(df['deltaH_decomposition (meV/atom)']))
print(df['tau'].corr(df['deltaG (meV/atom)']))
selected = df[(df['tau']>4.18) & (df['deltaG (meV/atom)']<0)]['Elements']
plt.yticks(**fontdict)
plt.axvline(x=4.18, color='red', linestyle='--', linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
x, y = df['tau'], df['deltaG (meV/atom)']
plt.ylim(-110, 110)

axs[1] = plt.subplot(1, 2, 2)
plt.hist(x=df['Band gap (eV)'], color='blue', edgecolor='black', linewidth=1.2)
plt.xlabel("Bandgap (eV)", **fontdict)
plt.xticks(**fontdict)
plt.ylabel(ylabel="Counts", **fontdict)
plt.yticks(**fontdict)
counts = df['Band type'].value_counts()
plt.text(1.1, 600, f"Indirect: {counts['Indirect']}", font='Times New Roman', size=30, fontweight='bold')
plt.text(1.1, 530, f"Non-indirect: {counts['Direct']+counts['Metallic']+counts['Semimetallic']}", font='Times New Roman', size=30, fontweight='bold')
fig.tight_layout()
print(len(df[(df['Band gap (eV)'] < 0.5)]))
print(len(df[(df['Band gap (eV)'] < 0.5) & (df['Band type'] == 'Indirect')]))

df.loc[df['Band type'] == "Indirect", 'Band type'] = 0
df.loc[df['Band type'] == "Direct", 'Band type'] = 1
df.loc[df['Band type'] == "Metallic", 'Band type'] = 1
df.loc[df['Band type'] == "Semimetallic", 'Band type'] = 1
df1 = pd.DataFrame(columns=['Cs_frac', 'K_frac', 'Rb_frac', 'Cd_frac', 'Ge_frac', 'Hg_frac', 'Pb_frac', 'Sn_frac', 'Zn_frac', 'Br_frac', 'Cl_frac', 'I_frac'])
for i in range(len(df)):
    A = df.iloc[i]['Elements'].split('.')[0]
    B1 = df.iloc[i]['Elements'].split('.')[1]
    B2 = df.iloc[i]['Elements'].split('.')[2]
    B3 = df.iloc[i]['Elements'].split('.')[3]
    B4 = df.iloc[i]['Elements'].split('.')[4]
    X = df.iloc[i]['Elements'].split('.')[5]
    Cs, K, Rb = 0, 0, 0
    if A == 'Cs':
        Cs = 1
    elif A =='K':
        K = 1
    else:
        Rb = 1
    Cd, Ge, Hg, Pb, Sn, Zn = 0, 0, 0, 0, 0, 0
    for B in [B1, B2, B3, B4]:
        if B == 'Cd':
            Cd += 0.25
        elif B =='Ge':
            Ge += 0.25
        elif B == 'Hg':
            Hg += 0.25
        elif B =='Pb':
            Pb += 0.25
        elif B =='Sn':
            Sn += 0.25
        elif B == 'Zn':
            Zn += 0.25
    Br, Cl, I = 0, 0, 0
    if X == 'Br':
        Br = 1
    elif X == 'Cl':
        Cl = 1
    else:
        I = 1
    df1 = pd.concat([df1, pd.DataFrame([[Cs, K, Rb, Cd, Ge, Hg, Pb, Sn, Zn, Br, Cl, I]], columns=df1.columns)], ignore_index=True)
df1['H_decomp'] = df['deltaH_decomposition (meV/atom)']
df1['tau'] = df['tau']
df.loc[df['Band gap (eV)'] < 0, 'Band gap (eV)'] = 0
print(len(df[df['Band gap (eV)']<0.5]))
df1['bandgap'] = df['Band gap (eV)']
df1['band type'] = df['Band type']
df1 = df1.astype(dtype=float)
corr = df1.corr()
corr = corr[['H_decomp', 'tau', 'bandgap', 'band type']]
corr = corr.iloc[:-4, :]
corr = corr.T
#fig, axs = plt.subplots(2, 2, figsize=(15, 6))
fig = plt.figure(figsize=(15, 6))
import seaborn as sns
ax = sns.heatmap(data=corr, annot=True, annot_kws = {'font': 'Times New Roman', 'size': 28, 'fontweight': 'bold'}, fmt = '.2f', linewidths=.5, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticklabels(['Cs_frac', 'K_frac', 'Rb_frac', 'Cd_frac', 'Ge_frac', 'Hg_frac', 'Pb_frac', 'Sn_frac', 'Zn_frac', 'Br_frac', 'Cl_frac', 'I_frac'], font='Times New Roman', size=28, fontweight='bold', rotation=45)
ax.set_yticklabels([r'$\mathregular{\Delta H_{decomp}}$', r'$\mathregular{\tau}$', 'Bandgap', 'Band type'], rotation=360, font='Times New Roman', size=28, fontweight='bold')
cbar = ax.collections[0].colorbar
cbar.set_label('Correlation coefficient', font='Times New Roman', size=24, fontweight='bold')
cbar.ax.tick_params(labelsize=20)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
fig.tight_layout()

plt.show()