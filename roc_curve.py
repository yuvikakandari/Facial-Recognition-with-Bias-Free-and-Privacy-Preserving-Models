from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Paste your real data here
y_true = [1, 1, 0, 0, 1]
y_scores = [0.85, 0.78, 0.30, 0.20, 0.90]

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Real Data)")
plt.legend()
plt.show()