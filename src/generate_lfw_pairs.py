import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from pathlib import Path

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

print("Dataset loaded successfully!")
print("Number of images:", len(images))
print("Number of identities:", len(lfw.target_names))

# Split images into reference and comparison sets.
indices = np.arange(len(images))

reference_idx, comparison_idx = train_test_split(
    indices,
    test_size=0.5,
    random_state=42,
    stratify=labels
)

# Group image indices by identity.
reference_by_person = {}
comparison_by_person = {}

for idx in reference_idx:
    person = labels[idx]
    reference_by_person.setdefault(person, []).append(idx)

for idx in comparison_idx:
    person = labels[idx]
    comparison_by_person.setdefault(person, []).append(idx)

# Keep only identities with images in both sets.
eligible_people = sorted(
    person
    for person in reference_by_person
    if person in comparison_by_person
)

rng = np.random.default_rng(42)

genuine_pairs = []
impostor_pairs = []

# Genuine pairs: two different images of the same person.
for person in eligible_people:
    ref = rng.choice(reference_by_person[person])
    comp = rng.choice(comparison_by_person[person])

    genuine_pairs.append((ref, comp))

# Impostor pairs: images belonging to different people.
for _ in range(len(genuine_pairs)):
    person_a, person_b = rng.choice(
        eligible_people,
        size=2,
        replace=False
    )

    ref = rng.choice(reference_by_person[person_a])
    comp = rng.choice(comparison_by_person[person_b])

    impostor_pairs.append((ref, comp))

# Convert lists into integer arrays.
genuine_pairs = np.array(genuine_pairs, dtype=np.int64)
impostor_pairs = np.array(impostor_pairs, dtype=np.int64)

# Save the pairs for reuse during evaluation.
np.save(
    RESULTS_DIR / "genuine_pairs.npy",
    genuine_pairs
)

np.save(
    RESULTS_DIR / "impostor_pairs.npy",
    impostor_pairs
)

print("\nPair generation complete!")
print("Number of genuine pairs:", len(genuine_pairs))
print("Number of impostor pairs:", len(impostor_pairs))

print("Example genuine pair:", genuine_pairs[0])
print("Example impostor pair:", impostor_pairs[0])

print("\nSaved files:")
print(RESULTS_DIR / "genuine_pairs.npy")
print(RESULTS_DIR / "impostor_pairs.npy")