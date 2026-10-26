import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_lfw_people

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

print("Loading LFW labels...")
lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)
labels = lfw.target

# Load validation and test pairs
val_genuine = np.load(RESULTS / "validation_genuine_pairs.npy")
val_impostor = np.load(RESULTS / "validation_impostor_pairs.npy")
test_genuine = np.load(RESULTS / "test_genuine_pairs.npy")
test_impostor = np.load(RESULTS / "test_impostor_pairs.npy")


def identities_in(pairs):
    return set(labels[pairs].ravel().tolist())


val_ids = identities_in(np.vstack([val_genuine, val_impostor]))
test_ids = identities_in(np.vstack([test_genuine, test_impostor]))

print("\nValidation identities used:", len(val_ids))
print("Test identities used:", len(test_ids))
print("Overlapping identities:", len(val_ids & test_ids))
print("Validation genuine pairs:", len(val_genuine))
print("Validation impostor pairs:", len(val_impostor))
print("Test genuine pairs:", len(test_genuine))
print("Test impostor pairs:", len(test_impostor))

# Verify identity separation
assert len(val_ids & test_ids) == 0, (
    "ERROR: Validation and test identities overlap."
)

# Verify genuine and impostor pair labels
assert np.all(labels[val_genuine[:, 0]] == labels[val_genuine[:, 1]])
assert np.all(labels[val_impostor[:, 0]] != labels[val_impostor[:, 1]])
assert np.all(labels[test_genuine[:, 0]] == labels[test_genuine[:, 1]])
assert np.all(labels[test_impostor[:, 0]] != labels[test_impostor[:, 1]])

print("\nPASS: Pair labels and identity separation are valid.")