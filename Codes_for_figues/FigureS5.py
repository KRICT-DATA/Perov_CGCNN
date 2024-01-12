import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df = pd.read_excel('bandgap_Exp_PBE0.xlsx')
fontdict = {'font': 'Times New Roman', 'size': 32, 'fontweight': 'bold'}
thickness = 2
fig, axes = plt.subplots(1, 1, figsize=(12,10))
ax1 = plt.subplot(1, 1, 1)
ax1.spines['left'].set_linewidth(thickness)
ax1.spines['right'].set_linewidth(thickness)
ax1.spines['top'].set_linewidth(thickness)
ax1.spines['bottom'].set_linewidth(thickness)
ax1.tick_params(length=5, width=thickness)
PBE0_rmse = np.sqrt(np.mean((df['Exp'] - df['PBE0'])**2))
HSE_rmse = np.sqrt(np.mean((df['Exp'] - df['HSE06'])**2))
B3LYP_rmse = np.sqrt(np.mean((df['Exp'] - df['B3LYP'])**2))
d1 = plt.scatter(x=df['Exp'], y=df['PBE0'], c='blue', alpha=0.75, s=250, label="PBE0")
d2 = plt.scatter(x=df['Exp'], y=df['HSE06'], c='orange', alpha=0.75, s=250, label="HSE06")
d3 = plt.scatter(x=df['Exp'], y=df['B3LYP'], c='black', alpha=0.75, s=250, label="B3LYP")
plt.plot([0.9,5], [0.9,5], color='grey', linestyle='--', linewidth=2)
plt.xlabel('Bandgap (eV), Exp.', **fontdict)
plt.ylabel('Bandgap (eV), DFT', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(0.9,5)
plt.xlim(0.9,5)
plt.text(2.0, 4.7, f"PBE0_RMSE = {PBE0_rmse: .3f}", font='Times New Roman', size=24, fontweight='bold', color='red')
plt.text(2.0, 4.4, f"HSE06_RMSE = {HSE_rmse: .3f}", font='Times New Roman', size=24, fontweight='bold', color='red')
plt.text(2.0, 4.1, f"B3LYP_RMSE = {B3LYP_rmse: .3f}", font='Times New Roman', size=24, fontweight='bold', color='red')
handles = [d1, d2, d3]
labels = [handle.get_label() for handle in handles]
legend = plt.legend(handles, labels, frameon=True, loc='upper left', framealpha=1.0, columnspacing=0.25, markerscale=1, handlelength=0.3, prop={'family': 'Times New Roman', 'size': 32, 'weight': 'bold'})
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('black')
plt.show()