
import numpy as np
import csv
from pathlib import Path
from PIL import Image, ImageFilter
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm

from embedding import FaceEmbedder

ROOT = Path(r"C:\face_recognition project")
RESULTS_DIR = ROOT / "results"

print("Loading LFW dataset...")

lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)

images = lfw.images
labels = lfw.target

embeddings = np.load(RESULTS_DIR / "lfw_embeddings.npy")

# Use the same reproducible sample as the brightness experiment
rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(images),
    size=min(200, len(images)),
    replace=False
)

embedder = FaceEmbedder()

# Gaussian blur radius: 0 means unchanged
blur_levels = [0, 1, 2, 3, 5, 7]
rows = []

print("\nRunning blur experiment...")

for idx in tqdm(sample_indices):
    image_array = np.clip(
        images[idx] * 255, 0, 255
    ).astype(np.uint8)

    original_image = Image.fromarray(image_array).convert("RGB")
    original_embedding = embeddings[idx]

    for radius in blur_levels:
        if radius == 0:
            altered_image = original_image
        else:
            altered_image = original_image.filter(
                ImageFilter.GaussianBlur(radius=radius)
            )

        altered_embedding = embedder.generate_embedding(
            altered_image
        ).cpu().numpy()[0]

        similarity = np.dot(
            original_embedding, altered_embedding
        ) / (
            np.linalg.norm(original_embedding)
            * np.linalg.norm(altered_embedding)
        )

        rows.append({
            "image_index": int(idx),
            "identity": int(labels[idx]),
            "blur_radius": radius,
            "cosine_similarity": float(similarity)
        })

output_csv = RESULTS_DIR / "lfw_blur_results.csv"

with open(output_csv, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_index",
            "identity",
            "blur_radius",
            "cosine_similarity"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nBlur experiment complete!")
print("Saved:", output_csv)

print("\nMean similarity by blur radius:")
for radius in blur_levels:
    values = [
        row["cosine_similarity"]
        for row in rows
        if row["blur_radius"] == radius
    ]

    print(f"Blur radius {radius}: {np.mean(values):.4f}")
