from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

distances = [3.2, 3.8, 4.1, 3.6, 4.0, 4.2]
labels    = [1,   1,   1,   0,   0,   0]

# Convert distance → score
scores = [-d for d in distances]

fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Real Model Data)")
plt.legend()
plt.show()