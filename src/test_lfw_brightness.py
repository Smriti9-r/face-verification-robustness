
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm
import csv

from embedding import FaceEmbedder

ROOT = Path(r"C:\face_recognition project")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the same LFW configuration used for baseline embeddings
print("Loading LFW dataset...")
lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)

images = lfw.images
labels = lfw.target

# Load original embeddings
embeddings = np.load(RESULTS_DIR / "lfw_embeddings.npy")

print("Images:", len(images))
print("Embedding shape:", embeddings.shape)

# Start with a manageable, reproducible sample
rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(images),
    size=min(200, len(images)),
    replace=False
)

embedder = FaceEmbedder()

brightness_levels = [1.0, 0.7, 0.5, 0.3, 0.15]
rows = []

print("\nRunning brightness experiment...")

for idx in tqdm(sample_indices):
    # LFW images are float RGB images in the range [0, 1]
    image_array = np.clip(images[idx] * 255, 0, 255).astype(np.uint8)
    original_image = Image.fromarray(image_array).convert("RGB")

    original_embedding = embeddings[idx:idx + 1]

    for level in brightness_levels:
        altered_image = ImageEnhance.Brightness(
            original_image
        ).enhance(level)

        altered_embedding = embedder.generate_embedding(
            altered_image
        ).cpu().numpy()

        # Cosine similarity between original and altered image embeddings
        a = original_embedding[0]
        b = altered_embedding[0]

        similarity = np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

        rows.append({
            "image_index": int(idx),
            "identity": int(labels[idx]),
            "brightness": level,
            "cosine_similarity": float(similarity)
        })

# Save all individual results
output_csv = RESULTS_DIR / "lfw_brightness_results.csv"

with open(output_csv, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_index",
            "identity",
            "brightness",
            "cosine_similarity"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nBrightness experiment complete!")
print("Saved:", output_csv)

print("\nMean similarity by brightness:")
for level in brightness_levels:
    values = [
        row["cosine_similarity"]
        for row in rows
        if row["brightness"] == level
    ]

    print(f"Brightness {level:.2f}: {np.mean(values):.4f}")
