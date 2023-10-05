# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:31:34 2023

@author: User
"""

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
df = pd.read_excel('Training data for CGCNN.xlsx')
df.to_csv('Data_CGCNN.csv', index=False)
lines = open('Data_CGCNN.csv').readlines()
#dH = [[],[],[],[]]
#dG = [[],[],[],[]]
newdat = []

for ll in lines[1:]:
    tmp = ll.split(',')
    c = tmp[0]
    v = float(tmp[1])
    
    b_sites = c.split('.')[1:-1]
    b_unique = list(set(b_sites))
    
    pi = np.array([b_sites.count(b)/4 for b in b_unique])
    dS_mix = -np.sum(pi*np.log(pi))
    
    nb = len(b_unique)
    newdat.append([nb,v,r'$\Delta H_\mathrm{decomp}$'])
    newdat.append([nb,v-kbT*dS_mix,r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'])
    
    
df = pd.DataFrame(newdat,columns=['Num','Energy','TdS'])
fontdict = {'font': 'Times New Roman', 'size': 24, 'fontweight': 'bold'}
fig = plt.figure(dpi=200, figsize=(12, 8))
plt.subplot(1,2,1)
ax = sns.violinplot(data=df,x='Num',y='Energy',hue='TdS',hue_order=[r'$\Delta H_\mathrm{decomp}$',r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'],split=True,cut=0)
ax.legend(handles=ax.legend_.legendHandles, labels=[r'$\Delta H_\mathrm{decomp}$',r'$\Delta H_\mathrm{decomp}-T\Delta S_\mathrm{mix}$'],fontsize=18)
plt.xlabel('The number of unique elements in B-site', **fontdict)
plt.ylabel('Energy (meV/atom)', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
#plt.title('(a) Energy distribution',fontsize=13)
plt.ylim(top=128)
lines = open('Data_sorted_CGCNN.csv').readlines()

y1 = []
y2 = []
y3 = []
elem = 'Ge'
for ll in lines[1:]:
    tmp = ll.split(',')
    c = tmp[0]
    v = float(tmp[-1])
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
#plt.title('(b) Effect of including Ge element',fontsize=13)

fig.tight_layout()
plt.show()
import os
os.remove('Data_CGCNN.csv')