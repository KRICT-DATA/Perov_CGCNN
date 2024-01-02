import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
fontdict = {'font': 'Times New Roman', 'size': 32, 'fontweight': 'bold'}
df = pd.read_excel('Training data for CGCNN.xlsx')
df1 = pd.read_csv('selected_validation.csv')
A = 'Cs'
X = 'Br'
B1 = 'Ge'
B2 = 'Sn'
k_B = 8.61736292496E-5
T = 298
thickness = 2
with open(f"./pickle_files_binary/{A}_{B1}{B2}_{X}.pkl", 'rb') as f:
    Results = pickle.load(f)
ratios, train_ratios, test_ratios = [], [], []
deltaHs, train_deltaHs = [], []
deltaGs, train_deltaGs, test_deltaGs = [], [], []
yerrs = []
for i in range(1, 16):
    ratio1 = float(i/16)
    ratio2 = 1-ratio1
    ratios.append(ratio1)
    deltaH = Results[f'{i}_{16-i}']['Prediction'][0]
    deltaHs.append(deltaH)
    max_deltaH = Results[f'{i}_{16 - i}']['Prediction'][1]
    yerrs.append(max_deltaH - deltaH)
    deltaS = 1000*k_B*T*(ratio1*np.log(ratio1) + ratio2*np.log(ratio2))
    deltaG = deltaH + deltaS
    deltaGs.append(deltaG)
    if i == 4:
        train_ratios.append(ratio1)
        train_deltaH = df[df['Elements'] == f'{A}.{B1}.{B2}.{B2}.{B2}.{X}']['deltaH_decomposition(meV/atom)']
        train_deltaGs.append(train_deltaH+deltaS)
    elif i == 8:
        train_ratios.append(ratio1)
        train_deltaH = df[df['Elements'] == f'{A}.{B1}.{B1}.{B2}.{B2}.{X}']['deltaH_decomposition(meV/atom)']
        train_deltaGs.append(train_deltaH + deltaS)
        train_ratios.append(ratio1)
        train_deltaH = df[df['Elements'] == f'{A}.{B1}.{B2}.{B1}.{B2}.{X}']['deltaH_decomposition(meV/atom)']
        train_deltaGs.append(train_deltaH + deltaS)
        train_ratios.append(ratio1)
        train_deltaH = df[df['Elements'] == f'{A}.{B1}.{B2}.{B2}.{B1}.{X}']['deltaH_decomposition(meV/atom)']
        train_deltaGs.append(train_deltaH + deltaS)
    elif i == 12:
        train_ratios.append(ratio1)
        train_deltaH = df[df['Elements'] == f'{A}.{B1}.{B1}.{B1}.{B2}.{X}']['deltaH_decomposition(meV/atom)']
        train_deltaGs.append(train_deltaH + deltaS)
test_ratios.append(0.75)
test_deltaGs.append(df1[df1['Compounds'] == f'{A}{B1}{0.75}{B2}{0.25}{X}3']['deltaG_DFT(meV/atom)'])

with open(f"./pickle_files_unary/{A}_{B2}_{X}.pkl", 'rb') as f:
    Results = pickle.load(f)
ratios.append(0), deltaGs.append(Results['16']['Prediction'][0]), yerrs.append(0)
train_ratios.append(0), train_deltaGs.append(df[df['Elements'] == f'{A}.{B2}.{B2}.{B2}.{B2}.{X}']['deltaH_decomposition(meV/atom)'])
with open(f"./pickle_files_unary/{A}_{B1}_{X}.pkl", 'rb') as f:
    Results = pickle.load(f)
ratios.append(1), deltaGs.append(Results['16']['Prediction'][0]), yerrs.append(0)
train_ratios.append(1), train_deltaGs.append(df[df['Elements'] == f'{A}.{B1}.{B1}.{B1}.{B1}.{X}']['deltaH_decomposition(meV/atom)'])

fig = plt.figure(figsize=(22, 11))
ax1 = plt.subplot(1, 2, 1)
ax1.spines['left'].set_linewidth(thickness)
ax1.spines['right'].set_linewidth(thickness)
ax1.spines['top'].set_linewidth(thickness)
ax1.spines['bottom'].set_linewidth(thickness)
ax1.tick_params(length=5, width=thickness)
yerr = np.hstack((np.zeros(len(yerrs)).reshape(1, -1), np.array(yerrs).reshape(1, -1))).reshape(2, len(yerrs))
d1 = plt.errorbar(ratios, deltaGs, yerr=yerr, fmt='o', markersize=10, color='black', alpha=0.25, label='CGCNN')
d2= plt.scatter(train_ratios, train_deltaGs, marker='s', s=75, color='red', alpha=0.75, label='Training data')
d3 = plt.scatter(test_ratios, test_deltaGs, marker='*', s=300,color='green', alpha=0.75, label='DFT data (80 atoms)')
plt.xlabel(f"x in {A}{B1}"r"$\mathregular{_x}$"f"{B2}"r"$\mathregular{_{1-x}}$"f"{X}"r"$\mathregular{_3}$",**fontdict)
plt.ylabel(r"$\mathregular{\Delta H_{decomp} -T\Delta S_{mix}}$ (meV/atom)", **fontdict)
plt.xticks(ticks=np.arange(0, 1.1, 0.1), font='Times New Roman', size=26, fontweight='bold')
plt.yticks(font='Times New Roman', size=28, fontweight='bold')
handles = [d1, d2, d3]
labels = [handle.get_label() for handle in handles]
legend = plt.legend(handles, labels, frameon=False, loc='upper right', framealpha=1.0, columnspacing=0.25, markerscale=1, handlelength=0.3, prop={'family': 'Times New Roman', 'size': 32, 'weight': 'bold'})
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('black')

def calc_entropy (ratios):
    entropy = 0
    for ratio in ratios:
        if ratio == 0.0:
            entropy += 0
        else:
            entropy += ratio*np.log(ratio)
    return entropy
A = 'Cs'
X = 'Cl'
B1 = "Ge"
B2 = 'Hg'
B3 = 'Sn'
x1, x2, x3, x_tr1, x_tr2, x_tr3 = [], [], [], [], [], []
G, G_tr = [], []
for i in range(17):
    ratio1 = float(i/16)
    for j in range(17-i):
        ratio2 = float(j/16)
        k = 16 - i - j
        ratio3 = float(k/16)
        x1.append(ratio1), x2.append(ratio2), x3.append(ratio3)
        deltaS = 1000 * k_B * T * calc_entropy([ratio1, ratio2, ratio3])
        if i==0 and j!=0 and k!=0:
            with open(f"./pickle_files_binary/{A}_{B2}{B3}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            deltaH = Results[f'{j}_{k}']['Prediction'][0]
            G.append(deltaH + deltaS)
        elif i!=0 and j==0 and k!=0:
            with open(f"./pickle_files_binary/{A}_{B1}{B3}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            deltaH = Results[f'{i}_{k}']['Prediction'][0]
            G.append(deltaH + deltaS)
        elif i!=0 and j!=0 and k==0:
            with open(f"./pickle_files_binary/{A}_{B1}{B2}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            deltaH = Results[f'{i}_{j}']['Prediction'][0]
            G.append(deltaH + deltaS)
        elif i!=0 and j!=0 and k!=0:
            with open(f"./pickle_files_ternary/{A}_{B1}{B2}{B3}_{X}.pkl", 'rb') as f:
                Results = pickle.load(f)
            deltaH = Results[f'{i}_{j}_{k}']['Prediction'][0]
            deltaG = deltaH + deltaS
            G.append(deltaG)
            if i ==4 and j ==4 and k==8:
                x_tr1.append(ratio1), x_tr2.append(ratio2), x_tr3.append(ratio3)
                deltaH_train1 = df[df['Elements'] == f"{A}.{B3}.{B3}.{B1}.{B2}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train2 = df[df['Elements'] == f"{A}.{B3}.{B1}.{B3}.{B2}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train3 = df[df['Elements'] == f"{A}.{B3}.{B1}.{B2}.{B3}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train = np.min([deltaH_train1, deltaH_train2, deltaH_train3])
                G_tr.append(deltaH_train+deltaS)
            elif i==4 and j==8 and k==4:
                x_tr1.append(ratio1), x_tr2.append(ratio2), x_tr3.append(ratio3)
                deltaH_train1 = df[df['Elements'] == f"{A}.{B2}.{B2}.{B1}.{B3}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train2 = df[df['Elements'] == f"{A}.{B2}.{B1}.{B2}.{B3}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train3 = df[df['Elements'] == f"{A}.{B2}.{B1}.{B3}.{B2}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train = np.min([deltaH_train1, deltaH_train2, deltaH_train3])
                G_tr.append(deltaH_train + deltaS)
            elif i==8 and j==4 and k==4:
                x_tr1.append(ratio1), x_tr2.append(ratio2), x_tr3.append(ratio3)
                deltaH_train1 = df[df['Elements'] == f"{A}.{B1}.{B1}.{B2}.{B3}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train2 = df[df['Elements'] == f"{A}.{B1}.{B2}.{B1}.{B3}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train3 = df[df['Elements'] == f"{A}.{B1}.{B2}.{B3}.{B1}.{X}"]['deltaH_decomposition(meV/atom)']
                deltaH_train = np.min([deltaH_train1, deltaH_train2, deltaH_train3])
                G_tr.append(deltaH_train + deltaS)
        else:
            #x_tr1.append(ratio1), x_tr2.append(ratio2), x_tr3.append(ratio3)
            G.append(0)#, G_tr.append(0)

G_val = float(df1[df1['Compounds'] == f"{A}{B1}0.5625{B2}0.25{B3}0.1875{X}3"]['deltaG_DFT(meV/atom)'].iloc[0])
vmin = np.min([np.min(G), np.min(G_tr)])
vmax = np.max([np.max(G), np.max(G_tr)])
#vmax = -80
#fig = plt.figure(figsize=(22, 11))
import mpltern
scatter_color = 'cyan'
linewidth = 2.0
ax2 = fig.add_subplot(1, 2, 2, projection='ternary')
cs = ax2.tripcolor(x1, x2, x3, G, cmap='gist_stern', vmin=vmin, vmax=vmax, shading='gouraud', label='CGCNN')
ax2.plot([0.5625, 0.5625], [0.0, 0.25], [0.4325, 0.1875], ls=':', color='cyan', linewidth=4)
ax2.plot([0.75, 0.5625], [0.25, 0.25], [0.0, 0.1875], ls=':', color='cyan', linewidth=4)
ax2.plot([0.0, 0.5625], [0.9, 0.25], [0.1875, 0.1875], ls=':', color='cyan', linewidth=4)
ax2.scatter(x_tr1, x_tr2, x_tr3, c=None, cmap=None, marker='s', s=100, edgecolor=scatter_color, linewidths=4, facecolor='none', vmin=vmin, vmax=vmax,alpha=1, label='Training data')
ax2.scatter(0.5625, 0.25, 0.1875, c=None, cmap=None,  marker='*', s=500, edgecolor=scatter_color, facecolor='none', linewidths=linewidth, alpha=1, label='Lowest')
print(G_val)
for i in range(len(x1)):
    if x1[i] == 0.5625 and x2[i] == 0.25 and x3[i] == 0.1875:
        print(G[i])
ax2.set_title(r"$\mathregular {CsGe_{x}Hg_{y}Sn_{1-x-y}Cl_3}$", font='Times New Roman', size=28, fontweight='bold', y=1.10)
ax2.set_tlabel(B1, **fontdict)
ax2.set_llabel(B2, **fontdict)
ax2.set_rlabel(B3, **fontdict)
position = 'tick1'
ax2.taxis.set_label_position(position)
ax2.laxis.set_label_position(position)
ax2.raxis.set_label_position(position)
ax2.tick_params(size=8, width=2)
ax2.taxis.set_ticklabels(ax2.taxis.get_ticklabels(), font='Times New Roman', size=26, weight='bold')
ax2.laxis.set_ticklabels(ax2.laxis.get_ticklabels(), font='Times New Roman', size=26, weight='bold')
ax2.raxis.set_ticklabels(ax2.raxis.get_ticklabels(), font='Times New Roman', size=26, weight='bold')
cax2 = ax2.inset_axes([1.1, 0.1, 0.05, 0.9], transform=ax2.transAxes)
colorbar1 = fig.colorbar(mappable=cs, cax=cax2)
colorbar1.set_label(r"$\mathregular{\Delta H_{decomp} -T\Delta S_{mix}}$ (meV/atom)", rotation=90, loc='center', va='baseline', labelpad=1, **fontdict)
colorbar1.ax.yaxis.set_label_coords(-0.6, 0.5)
colorbar1.ax.tick_params(length=4, width=2)
colorbar1.ax.set_yticklabels(colorbar1.ax.get_yticklabels(), font='Times New Roman', size=26, weight='bold')
from matplotlib.lines import Line2D
legend_marker = [Line2D([], [], marker='s', color=scatter_color, markeredgecolor=scatter_color, linewidth=linewidth),
Line2D([],[], marker='*', color=scatter_color, markeredgecolor=scatter_color, linewidth=linewidth, linestyle='')]
ax2.legend(handles=legend_marker, labels=['Training data', 'DFT data (80 atoms)'], frameon=False, framealpha=1.0, loc=(0.65, 0.9), handlelength=0.3, markerscale=3, prop={'family': 'Times New Roman', 'size': 24, 'weight': 'bold'})
plt.gcf().subplots_adjust(wspace=0.15)
fig.tight_layout()
plt.show()


