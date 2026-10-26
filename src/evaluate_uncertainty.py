
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load the identity-disjoint held-out test pairs.
genuine_pairs = np.load(RESULTS_DIR / "test_genuine_pairs.npy")
impostor_pairs = np.load(RESULTS_DIR / "test_impostor_pairs.npy")

embeddings = np.load(RESULTS_DIR / "lfw_embeddings.npy")

def cosine_scores(pairs):
    a = embeddings[pairs[:, 0]]
    b = embeddings[pairs[:, 1]]

    numerator = np.sum(a * b, axis=1)
    denominator = (
        np.linalg.norm(a, axis=1) *
        np.linalg.norm(b, axis=1)
    )
    return numerator / denominator

genuine_scores = cosine_scores(genuine_pairs)
impostor_scores = cosine_scores(impostor_pairs)

y_true = np.concatenate([
    np.ones(len(genuine_scores), dtype=int),
    np.zeros(len(impostor_scores), dtype=int)
])
scores = np.concatenate([genuine_scores, impostor_scores])

observed_auc = roc_auc_score(y_true, scores)

# Preliminary pair-level bootstrap.
# This is a rough uncertainty estimate, not identity-clustered inference.
rng = np.random.default_rng(42)
n_genuine = len(genuine_scores)
n_impostor = len(impostor_scores)
bootstrap_aucs = []

for _ in range(2000):
    sampled_genuine = rng.choice(
        genuine_scores, size=n_genuine, replace=True
    )
    sampled_impostor = rng.choice(
        impostor_scores, size=n_impostor, replace=True
    )

    sampled_scores = np.concatenate([
        sampled_genuine, sampled_impostor
    ])
    sampled_labels = np.concatenate([
        np.ones(n_genuine, dtype=int),
        np.zeros(n_impostor, dtype=int)
    ])

    bootstrap_aucs.append(
        roc_auc_score(sampled_labels, sampled_scores)
    )

lower, upper = np.percentile(bootstrap_aucs, [2.5, 97.5])

print("Held-out baseline AUC:", round(observed_auc, 4))
print("Preliminary 95% pair-bootstrap CI:", 
      f"[{lower:.4f}, {upper:.4f}]")
print("Genuine pairs:", n_genuine)
print("Impostor pairs:", n_impostor)
print(
    "Caution: pairs may share images; this interval may underestimate "
    "uncertainty and should be labelled preliminary."
)
