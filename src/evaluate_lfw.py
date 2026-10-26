from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_people
from PIL import Image

from embedding import FaceEmbedder

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading LFW dataset...")

lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)

images = lfw.images
labels = lfw.target

print("Images loaded:", len(images))

embedder = FaceEmbedder()
embeddings = []

print("Generating embeddings...")

for i, image_array in enumerate(images):
    # Convert RGB pixel values from [0, 1] to uint8 [0, 255].
    image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    face_image = Image.fromarray(image_array)

    embedding = embedder.generate_embedding(face_image)
    embeddings.append(embedding.squeeze(0).cpu().numpy())

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(images)} images")

embeddings = np.asarray(embeddings, dtype=np.float32)

# Save embeddings and identity labels
embeddings_path = RESULTS_DIR / "lfw_embeddings.npy"
labels_path = RESULTS_DIR / "lfw_labels.npy"

np.save(embeddings_path, embeddings)
np.save(labels_path, labels)

print("\nEmbedding generation complete!")
print("Embedding matrix shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print(f"Saved embeddings to: {embeddings_path}")
print(f"Saved labels to: {labels_path}")