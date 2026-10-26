from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load face embeddings and generated pairs
embeddings = np.load(RESULTS_DIR / "lfw_embeddings.npy")
genuine_pairs = np.load(RESULTS_DIR / "genuine_pairs.npy")
impostor_pairs = np.load(RESULTS_DIR / "impostor_pairs.npy")


def calculate_scores(pairs):
    scores = []

    for idx_a, idx_b in pairs:
        score = cosine_similarity(
            embeddings[[idx_a]],
            embeddings[[idx_b]]
        )[0, 0]

        scores.append(score)

    return np.asarray(scores, dtype=np.float32)


# Calculate similarity scores
genuine_scores = calculate_scores(genuine_pairs)
impostor_scores = calculate_scores(impostor_pairs)

# Display summary statistics
print("\nEvaluation complete!")
print("Embedding matrix shape:", embeddings.shape)
print("Genuine pairs:", len(genuine_scores))
print("Impostor pairs:", len(impostor_scores))

print("\nGenuine similarity scores:")
print("Mean:", round(float(genuine_scores.mean()), 4))
print("Minimum:", round(float(genuine_scores.min()), 4))
print("Maximum:", round(float(genuine_scores.max()), 4))

print("\nImpostor similarity scores:")
print("Mean:", round(float(impostor_scores.mean()), 4))
print("Minimum:", round(float(impostor_scores.min()), 4))
print("Maximum:", round(float(impostor_scores.max()), 4))

# Save scores for the next evaluation stage
np.save(RESULTS_DIR / "genuine_scores.npy", genuine_scores)
np.save(RESULTS_DIR / "impostor_scores.npy", impostor_scores)

print("\nSaved genuine_scores.npy and impostor_scores.npy")