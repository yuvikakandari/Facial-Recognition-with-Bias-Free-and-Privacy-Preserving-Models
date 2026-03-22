from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = [1, 1, 0, 0]  # 1 = genuine, 0 = imposter
y_scores = [0.9, 0.6, 0.4, 0.2]  # confidence scores

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()