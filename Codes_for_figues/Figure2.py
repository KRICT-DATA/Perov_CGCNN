import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 13})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

kb = 8.6173e-2  #meV K-1
T = 298
kbT = kb*T
df = pd.read_excel('../Data/Training data for CGCNN.xlsx')
df = df.rename(columns={'Elements (A.B1.n1.B2.n2.B3.n3.B4.n4.X)': 'Elements'})
newdat = []

for idx, row in df.iterrows():
    c = row['Elements']
    v = float(row['deltaH_decomposition (meV/atom)'])
    
    b_sites = c.split('.')[1:-1]
    b_unique = list(set(b_sites))
    
    pi = np.array([b_sites.count(b)/4 for b in b_unique])
    dS_mix = -np.sum(pi*np.log(pi))
    
    nb = len(b_unique)
    newdat.append([nb,v,r'$\Delta H_\mathrm{decomp}$'])
    newdat.append([nb,v-kbT*dS_mix,r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'])
    
    
df1 = pd.DataFrame(newdat,columns=['Num','Energy','TdS'])
fontdict = {'font': 'Times New Roman', 'size': 24, 'fontweight': 'bold'}
fig = plt.figure(dpi=200, figsize=(12, 8))
plt.subplot(1,2,1)
ax = sns.violinplot(data=df1,x='Num',y='Energy',hue='TdS',hue_order=[r'$\Delta H_\mathrm{decomp}$',r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'],split=True,cut=0)
ax.legend(handles=ax.legend_.legendHandles, labels=[r'$\Delta H_\mathrm{decomp}$',r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'],fontsize=18)
plt.xlabel('The number of unique elements in B-site', **fontdict)
plt.ylabel('Energy (meV/atom)', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(top=128)


y1 = []
y2 = []
y3 = []
elem = 'Ge'
for idx, row in df.iterrows():
    c = row['Elements']
    v = float(row['deltaH_decomposition (meV/atom)'])
    y1.append(v)
    if elem in c:
        y2.append(v)
    else:
        y3.append(v)

plt.subplot(1,2,2)
sns.kdeplot(y2,color='red',label=f'With {elem}')
sns.kdeplot(y3,color='green',label=f'Without {elem}')
plt.legend(loc='upper right',prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'})

plt.axvline(np.mean(y2),ls='--',color='red')
plt.axvline(np.mean(y3),ls='--',color='green')

plt.ylabel('Density', **fontdict)
plt.xlabel(r'$\mathregular {\Delta H_{decomp}}$ (meV/atom)', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)

fig.tight_layout()
plt.show()
import os
os.remove('Data_CGCNN.csv')