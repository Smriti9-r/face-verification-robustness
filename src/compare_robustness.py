import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load brightness and blur results
brightness = pd.read_csv(
    RESULTS_DIR / "brightness_verification_metrics.csv"
)
blur = pd.read_csv(
    RESULTS_DIR / "blur_verification_metrics.csv"
)

# Standardize column names
brightness = brightness.rename(
    columns={"brightness": "severity"}
)
brightness["condition"] = "brightness"

blur = blur.rename(
    columns={"blur_radius": "severity"}
)
blur["condition"] = "blur"

# Combine results
combined = pd.concat(
    [brightness, blur],
    ignore_index=True
)

# Select and order columns
combined = combined[
    [
        "condition",
        "severity",
        "auc",
        "eer",
        "far_at_baseline_threshold",
        "frr_at_baseline_threshold"
    ]
]

# Save combined results
output = RESULTS_DIR / "combined_robustness_metrics.csv"
combined.to_csv(output, index=False)

print("\nCombined robustness results:")
print(combined.to_string(index=False))
print("\nSaved:", output)