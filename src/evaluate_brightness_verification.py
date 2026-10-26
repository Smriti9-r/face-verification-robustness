
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from embedding import FaceEmbedder

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load the same LFW image ordering.
lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)
images = lfw.images

# Use held-out test pairs only.
genuine_pairs = np.load(
    RESULTS_DIR / "test_genuine_pairs.npy"
)
impostor_pairs = np.load(
    RESULTS_DIR / "test_impostor_pairs.npy"
)

pairs = np.vstack([genuine_pairs, impostor_pairs])

y_true = np.concatenate([
    np.ones(len(genuine_pairs), dtype=int),
    np.zeros(len(impostor_pairs), dtype=int)
])

# Selected using validation data only.
baseline_threshold = 0.4085

brightness_levels = [1.0, 0.7, 0.5, 0.3, 0.15]

embedder = FaceEmbedder()
original_embeddings = np.load(
    RESULTS_DIR / "lfw_embeddings.npy"
)

results = []

for factor in brightness_levels:
    print(f"\nEvaluating brightness factor {factor}...")

    altered_cache = {}

    # Alter only the second image in each pair.
    for idx in tqdm(np.unique(pairs[:, 1])):
        idx = int(idx)

        if factor == 1.0:
            altered_cache[idx] = original_embeddings[idx]
            continue

        arr = np.clip(
            images[idx] * 255, 0, 255
        ).astype(np.uint8)

        image = Image.fromarray(arr).convert("RGB")
        altered = ImageEnhance.Brightness(image).enhance(factor)

        altered_cache[idx] = (
            embedder.generate_embedding(altered)
            .cpu().numpy()[0]
        )

    scores = []

    for idx_a, idx_b in pairs:
        a = original_embeddings[idx_a]
        b = altered_cache[int(idx_b)]

        score = np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )
        scores.append(float(score))

    scores = np.asarray(scores)

    auc = roc_auc_score(y_true, scores)

    # Descriptive EER; this sweeps thresholds on test data.
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # FAR and FRR at the fixed validation threshold.
    predicted_match = scores >= baseline_threshold
    far = np.mean(predicted_match[y_true == 0])
    frr = np.mean(~predicted_match[y_true == 1])

    results.append({
        "brightness_factor": factor,
        "auc": float(auc),
        "eer": float(eer),
        "far_at_validation_threshold": float(far),
        "frr_at_validation_threshold": float(frr)
    })

output = RESULTS_DIR / "heldout_brightness_metrics.csv"

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\nHeld-out brightness verification results")
print("Validation-selected threshold:", baseline_threshold)

for row in results:
    print(
        f"Brightness {row['brightness_factor']}: "
        f"AUC={row['auc']:.4f}, "
        f"EER={row['eer']:.4f}, "
        f"FAR={row['far_at_validation_threshold']:.4f}, "
        f"FRR={row['frr_at_validation_threshold']:.4f}"
    )

print("\nSaved:", output)
