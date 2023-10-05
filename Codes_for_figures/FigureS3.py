import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df = pd.read_excel('Selected.xlsx')
fontdict = {'font': 'Times New Roman', 'size': 32, 'fontweight': 'bold'}
thickness = 2
#fig, axes = plt.subplots(1, 3, figsize=(26,10), gridspec_kw={'width_ratios': [1, 1, 0.5], 'wspace': 0.5})
fig, axes = plt.subplots(1, 2, figsize=(18,9))
ax1 = plt.subplot(1, 2, 1)
ax1.spines['left'].set_linewidth(thickness)
ax1.spines['right'].set_linewidth(thickness)
ax1.spines['top'].set_linewidth(thickness)
ax1.spines['bottom'].set_linewidth(thickness)
ax1.tick_params(length=5, width=thickness)
y_true = df.iloc[:, 2]
y_pred = df.iloc[:, 1]
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)
plt.scatter(x=y_true, y=y_pred, c='black', alpha=0.5)
plt.plot([-500, 150], [-500, 150], color='red', linestyle='--', linewidth=2)
plt.xlabel(r'$\mathregular{\Delta H_{decomp}-T\Delta S_{mix}}$' ' (meV/atom), DFT', **fontdict)
plt.ylabel(r'$\mathregular{\Delta H_{decomp}-T\Delta S_{mix}}$' ' (meV/atom), CGCNN', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(-120,-20)
plt.xlim(-120,-20)
plt.text(-110, -30, f"MAE = {mae: .3f}", font='Times New Roman', size=38, fontweight='bold')
plt.text(-110, -40, r"$\mathregular{R^2}$"f" = {r2:.3f}",font='Times New Roman', size=38, fontweight='bold')
plt.tick_params(width=2)

ax2 = plt.subplot(1, 2, 2)
ax2.spines['left'].set_linewidth(thickness)
ax2.spines['right'].set_linewidth(thickness)
ax2.spines['top'].set_linewidth(thickness)
ax2.spines['bottom'].set_linewidth(thickness)
ax2.tick_params(length=5, width=thickness)
df.loc[df['Band gap_CGCNN'] < 0, 'Band gap_CGCNN'] = 0
df.loc[df['Band gap_DFT'] < 0, 'Band gap_DFT'] = 0
y_true = df.iloc[:, 6]
y_pred = df.iloc[:, 5]
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)
plt.scatter(x=y_true, y=y_pred, c='black', alpha=0.5)
plt.plot([-1, 2.7], [-1, 2.7], color='red', linestyle='--', linewidth=2)
plt.xlabel('Band gap (eV), DFT', **fontdict)
plt.ylabel('Band gap (eV), CGCNN', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(-0.2,1.0)
plt.xlim(-0.2,1.0)
plt.text(-0.1, 0.9, f"MAE = {mae: .3f}", font='Times New Roman', size=38, fontweight='bold')
plt.text(-0.1, 0.75, r"$\mathregular{R^2}$" f" = {r2: .3f}", font='Times New Roman', size=38, fontweight='bold')

fig.tight_layout()
plt.show()

ax2 = plt.subplot(1, 2, 2)
ax2.spines['left'].set_linewidth(thickness)
ax2.spines['right'].set_linewidth(thickness)
ax2.spines['top'].set_linewidth(thickness)
ax2.spines['bottom'].set_linewidth(thickness)
ax2.tick_params(length=5, width=thickness)
y_pred = df[df['Band type_DFT'] == 'Direct']['Band gap_CGCNN']
y_true = df[df['Band type_DFT'] == 'Direct']['Band gap_DFT']
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)
plt.scatter(x=y_true, y=y_pred, c='black', alpha=0.5)
plt.plot([-1, 2.7], [-1, 2.7], color='red', linestyle='--', linewidth=2)
plt.xlabel('Band gap (eV), DFT', **fontdict)
plt.ylabel('Band gap (eV), CGCNN', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(-0.2,1.0)
plt.xlim(-0.2,1.0)
plt.text(-0.1, 0.9, f"MAE = {mae: .3f}", font='Times New Roman', size=38, fontweight='bold')
plt.text(-0.1, 0.75, r"$\mathregular{R^2}$" f" = {r2: .3f}", font='Times New Roman', size=38, fontweight='bold')

plt.show()

ax3 = plt.subplot(1, 3, 3)
ax3.spines['left'].set_linewidth(thickness)
ax3.spines['right'].set_linewidth(thickness)
ax3.spines['top'].set_linewidth(thickness)
ax3.spines['bottom'].set_linewidth(thickness)
ax3.tick_params(length=5, width=thickness)
print(len(df[df['Band type_DFT'] == "Indirect"]))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = []
for i in range(len(df)):
    if df.iloc[i, 4] == "Indirect":
        y_true.append(0)
    else:
        y_true.append(1)
cm = confusion_matrix(y_true=y_true, y_pred=[1 for i in range(len(df))], labels=[0, 1])
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print(precision, recall, accuracy, f1_score)

disp = ConfusionMatrixDisplay(cm, display_labels=['Indirect', 'Non-indirect'])
#disp.plot()
labels = ['Indirect', 'Non-\nindirect']
im = ax3.imshow(cm, cmap='gray', alpha=0, vmin=0, vmax=1, extent=[-0.5, len(labels) - 0.5, -0.5, len(labels) - 0.5])
tick_marks = np.arange(len(labels))
ax3.set_title(f"Accuracy = {accuracy: .2f}\n", **fontdict)
ax3.set_xticks(tick_marks)
ax3.set_xticklabels(labels, **fontdict)
plt.xlabel("Predicted labels", **fontdict)
ax3.set_yticks(tick_marks)
ax3.set_yticklabels(labels, **fontdict)
plt.ylabel("True labels", **fontdict)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax3.text(j, i, str(cm[i, j]), va='center', ha='center', **fontdict)
ax3.hlines(np.arange(len(labels)) - 0.5, -0.5, len(labels) - 0.5, colors='black', linewidths=1)
ax3.vlines(np.arange(len(labels)) - 0.5, -0.5, len(labels) - 0.5, colors='black', linewidths=1)

