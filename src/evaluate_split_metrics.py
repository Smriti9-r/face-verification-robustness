
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

embeddings = np.load(RESULTS_DIR / "lfw_embeddings.npy")


def pair_scores(pair_indices):
    """Calculate cosine similarity for each pair of embeddings."""
    first = embeddings[pair_indices[:, 0]]
    second = embeddings[pair_indices[:, 1]]

    numerator = np.sum(first * second, axis=1)
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
    )

    return numerator / np.maximum(denominator, 1e-12)


def load_scores(prefix):
    genuine_pairs = np.load(
        RESULTS_DIR / f"{prefix}_genuine_pairs.npy"
    )
    impostor_pairs = np.load(
        RESULTS_DIR / f"{prefix}_impostor_pairs.npy"
    )

    genuine_scores = pair_scores(genuine_pairs)
    impostor_scores = pair_scores(impostor_pairs)

    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores), dtype=int),
        np.zeros(len(impostor_scores), dtype=int)
    ])

    return genuine_scores, impostor_scores, scores, labels


# 1. Load validation and test scores.
val_genuine, val_impostor, val_scores, val_labels = load_scores(
    "validation"
)
test_genuine, test_impostor, test_scores, test_labels = load_scores(
    "test"
)

# 2. Choose threshold using validation data only.
val_fpr, val_tpr, val_thresholds = roc_curve(
    val_labels, val_scores
)
val_fnr = 1 - val_tpr

valid = np.isfinite(val_thresholds)
eer_index = np.argmin(
    np.abs(val_fpr[valid] - val_fnr[valid])
)

threshold = val_thresholds[valid][eer_index]

# 3. Evaluate the selected threshold on the held-out test set.
test_auc = roc_auc_score(test_labels, test_scores)
test_fpr, test_tpr, test_thresholds = roc_curve(
    test_labels, test_scores
)
test_fnr = 1 - test_tpr

test_eer_index = np.argmin(np.abs(test_fpr - test_fnr))
test_eer = (test_fpr[test_eer_index] + test_fnr[test_eer_index]) / 2

# Higher similarity means "same person".
predicted_same = test_scores >= threshold
actual_same = test_labels == 1

far = np.mean(predicted_same[~actual_same])
frr = np.mean(~predicted_same[actual_same])

print("\nVALIDATION")
print("Genuine pairs:", len(val_genuine))
print("Impostor pairs:", len(val_impostor))
print(f"Selected threshold: {threshold:.4f}")

print("\nHELD-OUT TEST")
print("Genuine pairs:", len(test_genuine))
print("Impostor pairs:", len(test_impostor))
print(f"AUC: {test_auc:.4f}")
print(f"EER: {test_eer:.4f}")
print(f"FAR at validation threshold: {far:.4f}")
print(f"FRR at validation threshold: {frr:.4f}")

output = RESULTS_DIR / "heldout_test_metrics.csv"

with open(output, "w", encoding="utf-8", newline="") as file:
    file.write(
        "threshold,test_auc,test_eer,far_at_validation_threshold,"
        "frr_at_validation_threshold\n"
    )
    file.write(
        f"{threshold:.6f},{test_auc:.6f},{test_eer:.6f},"
        f"{far:.6f},{frr:.6f}\n"
    )

print("\nSaved:", output)
