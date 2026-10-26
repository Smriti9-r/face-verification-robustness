
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

brightness_file = RESULTS_DIR / "heldout_brightness_metrics.csv"
blur_file = RESULTS_DIR / "heldout_blur_metrics.csv"
output_file = RESULTS_DIR / "combined_heldout_robustness.csv"

combined_rows = []

# Read brightness results.
with open(brightness_file, "r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        row["condition"] = "brightness"
        row["severity"] = row.pop("brightness_factor")
        combined_rows.append(row)

# Read blur results.
with open(blur_file, "r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        row["condition"] = "blur"
        row["severity"] = row.pop("blur_radius")
        combined_rows.append(row)

columns = [
    "condition",
    "severity",
    "auc",
    "eer",
    "far_at_validation_threshold",
    "frr_at_validation_threshold",
]

# Write the combined results.
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(combined_rows)

print("\nCombined held-out robustness results:")
for row in combined_rows:
    print(
        f"{row['condition']:10} "
        f"severity={row['severity']:>5} "
        f"AUC={float(row['auc']):.4f} "
        f"EER={float(row['eer']):.4f} "
        f"FAR={float(row['far_at_validation_threshold']):.4f} "
        f"FRR={float(row['frr_at_validation_threshold']):.4f}"
    )

print("\nSaved:", output_file)
