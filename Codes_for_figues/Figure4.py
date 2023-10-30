import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
fontdict = {'font': 'Times New Roman', 'size': 32, 'fontweight': 'bold'}
thickness = 2
fig, axes = plt.subplots(1, 3, figsize=(26,10), gridspec_kw={'width_ratios': [1, 1, 0.5], 'wspace': 0.5})
ax1 = plt.subplot(1, 3, 1)
ax1.spines['left'].set_linewidth(thickness)
ax1.spines['right'].set_linewidth(thickness)
ax1.spines['top'].set_linewidth(thickness)
ax1.spines['bottom'].set_linewidth(thickness)
ax1.tick_params(length=5, width=thickness)
df = pd.read_csv('test_results_deltaH_decomp.csv', header=None)
y_true = df.iloc[:, 1]
y_pred = df.iloc[:, 2]
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)
plt.scatter(x=y_true, y=y_pred, c='black', alpha=0.5)
#plt.plot([np.min(y_true)-np.mean(y_true), np.max(y_true)+np.mean(y_true)], [np.min(y_true)-np.mean(y_true), np.max(y_true)+np.mean(y_true)], color='red', linestyle='--', linewidth=2)
plt.plot([-500, 150], [-500, 150], color='red', linestyle='--', linewidth=2)
plt.xlabel(r'$\mathregular{\Delta H_{decomp}}$' ' (meV/atom), DFT', **fontdict)
plt.ylabel(r'$\mathregular{\Delta H_{decomp}}$' ' (meV/atom), CGCNN', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(-100,100)
plt.xlim(-100,100)
plt.text(-90, 75, f"MAE = {mae: .3f}", font='Times New Roman', size=38, fontweight='bold')
plt.text(-90, 50, r"$\mathregular{R^2}$"f" = {r2:.3f}",font='Times New Roman', size=38, fontweight='bold')
plt.tick_params(width=2)

ax2 = plt.subplot(1, 3, 2)
ax2.spines['left'].set_linewidth(thickness)
ax2.spines['right'].set_linewidth(thickness)
ax2.spines['top'].set_linewidth(thickness)
ax2.spines['bottom'].set_linewidth(thickness)
ax2.tick_params(length=5, width=thickness)
'''
error = np.abs(y_true - y_pred)
plt.hist(x=error, color='black', alpha=0.5, edgecolor='black',)
plt.xlabel("Error (meV/atom)", **fontdict)
plt.ylabel("Counts", **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
'''
df = pd.read_csv('test_result_bandgap.csv', header=None)
y_true = df.iloc[:, 1]
y_pred = df.iloc[:, 2]
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean(np.abs(y_true - y_pred)**2))
print(rmse)
r2 = r2_score(y_true, y_pred)
plt.scatter(x=y_true, y=y_pred, c='black', alpha=0.5)
plt.plot([-1, 2.7], [-1, 2.7], color='red', linestyle='--', linewidth=2)
plt.xlabel('Band gap (eV), DFT', **fontdict)
plt.ylabel('Band gap (eV), CGCNN', **fontdict)
plt.xticks(**fontdict)
plt.yticks(**fontdict)
plt.ylim(-0.5,2.7)
plt.xlim(-0.5,2.7)
plt.text(-0.25, 2.3, f"MAE = {mae: .3f}", font='Times New Roman', size=38, fontweight='bold')
plt.text(-0.25, 1.8, r"$\mathregular{R^2}$" f" = {r2: .3f}", font='Times New Roman', size=38, fontweight='bold')

ax3 = plt.subplot(1, 3, 3)
ax3.spines['left'].set_linewidth(thickness)
ax3.spines['right'].set_linewidth(thickness)
ax3.spines['top'].set_linewidth(thickness)
ax3.spines['bottom'].set_linewidth(thickness)
ax3.tick_params(length=5, width=thickness)
df = pd.read_csv('test_result_bandtype.csv', header=None)
for i in range(len(df)):
    if df.iloc[i, 2] < 0.5:
        df.iloc[i, 2] = 0
    else:
        df.iloc[i, 2] = 1
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true=df.iloc[:, 1], y_pred=df.iloc[:, 2], labels=[0, 1])
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
#ax2.imshow(cm)
#import seaborn as sns
#sns.heatmap(cm)
disp = ConfusionMatrixDisplay(cm, display_labels=['Indirect', 'Non-indirect'])
#disp.plot()
labels = ['Indirect', 'Non-\nindirect']
im = ax3.imshow(cm, cmap='gray', alpha=0, vmin=0, vmax=1, extent=[-0.5, len(labels) - 0.5, -0.5, len(labels) - 0.5])
tick_marks = np.arange(len(labels))
ax3.set_title(f"Accuracy = {accuracy: .2f}\nPrecision = {precision: .2f}\nRecall = {recall: .2f}\nF1 score = {f1_score: .2f}", **fontdict)
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
fig.tight_layout()
#plt.colorbar(im, ax=ax3)
plt.show()