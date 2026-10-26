import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load ROC results
data = np.load(RESULTS_DIR / "roc_results.npz")

fpr = data["fpr"]
tpr = data["tpr"]
auc = float(data["auc"])
eer = float(data["eer"])

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC curve (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Acceptance Rate (FAR)")
plt.ylabel("True Acceptance Rate (1 - FRR)")
plt.title("Face Verification ROC Curve on LFW Pairs")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = RESULTS_DIR / "roc_curve.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nROC curve saved to: {output_path}")
print(f"AUC: {auc:.4f}")
print(f"EER: {eer * 100:.2f}%")