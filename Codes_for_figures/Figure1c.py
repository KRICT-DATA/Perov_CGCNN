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
#df = pd.read_csv('Data_CGCNN.csv')
df = pd.read_excel('Training data for CGCNN.xlsx')
entropy = []
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
df['deltaG (meV/atom)'] = df['deltaH_decomposition(meV/atom)'] + entropy
idx_min = df['deltaH_decomposition(meV/atom)'].idxmin()
print(df.iloc[idx_min]['Elements'])
idx_min = df['deltaG (meV/atom)'].idxmin()
print(df.iloc[idx_min]['Elements'])
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1)
fig.set_size_inches((11, 9))
fontdict = {'font': 'Times New Roman', 'size': 36, 'fontweight': 'bold'}
thickness = 2
axes.spines['left'].set_linewidth(thickness)
axes.spines['right'].set_linewidth(thickness)
axes.spines['top'].set_linewidth(thickness)
axes.spines['bottom'].set_linewidth(thickness)
axes.tick_params(length=5, width=thickness)
axes = plt.subplot(1,1,1)
num_bins = 30
bin_range = (-105, 100)
plt.hist(x=df['deltaH_decomposition(meV/atom)'], color='red', label=r"$\mathregular {\Delta H_{decomp}}$", edgecolor='black', linewidth=1.2, bins=num_bins, range=bin_range)
plt.hist(x=df['deltaG (meV/atom)'], color='green', label=r"$\mathregular {\Delta H_{decomp}-T\Delta S_{mix}}$", edgecolor='black', linewidth=1.2, alpha=0.25, bins=num_bins, range=bin_range)
plt.xlabel("Energy (meV/atom)", **fontdict)
plt.ylabel(ylabel="Counts", **fontdict)
plt.xticks(font='Times New Roman', size=28, fontweight='bold')
plt.yticks(ticks=np.arange(0, 450, 50), font='Times New Roman', size=36, fontweight='bold')
plt.legend(frameon=False, prop={'family': 'Times New Roman', 'size': 36, 'weight': 'bold'}, loc='upper right', columnspacing=0.5, handlelength=1.0,)
fig.tight_layout()
plt.show()