from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load genuine and impostor similarity scores
genuine_scores = np.load(RESULTS_DIR / "genuine_scores.npy")
impostor_scores = np.load(RESULTS_DIR / "impostor_scores.npy")

# 1 = same person, 0 = different people
y_true = np.concatenate([
    np.ones(len(genuine_scores)),
    np.zeros(len(impostor_scores))
])

scores = np.concatenate([
    genuine_scores,
    impostor_scores
])

# Calculate ROC values and AUC
fpr, tpr, thresholds = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)

# False rejection rate
fnr = 1 - tpr

# Find the threshold where FAR and FRR are closest
eer_index = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_index] + fnr[eer_index]) / 2
eer_threshold = thresholds[eer_index]

print("\nVerification metrics")
print("--------------------")
print(f"AUC: {auc:.4f}")
print(f"EER: {eer:.4f} ({eer * 100:.2f}%)")
print(f"Approximate EER threshold: {eer_threshold:.4f}")

# Save ROC data for plotting later
np.savez(
    RESULTS_DIR / "roc_results.npz",
    fpr=fpr,
    tpr=tpr,
    thresholds=thresholds,
    auc=auc,
    eer=eer,
    eer_threshold=eer_threshold
)

print("\nSaved ROC data to results/roc_results.npz")