
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

labels = lfw.target
indices = np.arange(len(labels))

# Split identities, not individual images.
people = np.unique(labels)

validation_people, test_people = train_test_split(
    people,
    test_size=0.5,
    random_state=42
)

rng = np.random.default_rng(42)


def make_pairs(selected_people, labels, rng):
    """Create balanced genuine and impostor pairs from selected identities."""

    by_person = {}

    for idx, person in enumerate(labels):
        if person in selected_people:
            by_person.setdefault(person, []).append(idx)

    eligible = [
        person for person, image_ids in by_person.items()
        if len(image_ids) >= 2
    ]

    genuine_pairs = []
    impostor_pairs = []

    for person in eligible:
        image_ids = rng.choice(
            by_person[person], size=2, replace=False
        )
        genuine_pairs.append(image_ids)

    # Create one impostor pair per genuine pair.
    for _ in range(len(genuine_pairs)):
        person_a, person_b = rng.choice(
            eligible, size=2, replace=False
        )

        image_a = rng.choice(by_person[person_a])
        image_b = rng.choice(by_person[person_b])

        impostor_pairs.append((image_a, image_b))

    return (
        np.asarray(genuine_pairs, dtype=np.int64),
        np.asarray(impostor_pairs, dtype=np.int64)
    )


val_genuine, val_impostor = make_pairs(
    set(validation_people), labels, rng
)

test_genuine, test_impostor = make_pairs(
    set(test_people), labels, rng
)

np.save(RESULTS_DIR / "validation_genuine_pairs.npy", val_genuine)
np.save(RESULTS_DIR / "validation_impostor_pairs.npy", val_impostor)
np.save(RESULTS_DIR / "test_genuine_pairs.npy", test_genuine)
np.save(RESULTS_DIR / "test_impostor_pairs.npy", test_impostor)

print("\nPair generation complete.")
print("Validation identities:", len(validation_people))
print("Test identities:", len(test_people))
print("Validation genuine pairs:", len(val_genuine))
print("Validation impostor pairs:", len(val_impostor))
print("Test genuine pairs:", len(test_genuine))
print("Test impostor pairs:", len(test_impostor))
print("\nSaved validation and test pair files in:", RESULTS_DIR)
