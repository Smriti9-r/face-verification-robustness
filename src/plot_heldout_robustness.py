import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Create figures directory if it doesn't exist
FIGURES.mkdir(parents=True, exist_ok=True)

# Load combined held-out metrics
data = []
csv_file = RESULTS / "combined_heldout_robustness.csv"

with open(csv_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        row["severity"] = float(row["severity"])
        row["auc"] = float(row["auc"])
        row["frr_at_validation_threshold"] = float(
            row["frr_at_validation_threshold"]
        )
        data.append(row)

# Check that both experimental conditions are present
conditions = {row["condition"] for row in data}

if not data:
    raise ValueError(f"No data found in {csv_file}")

if not {"brightness", "blur"}.issubset(conditions):
    raise ValueError(
        "Expected both 'brightness' and 'blur' conditions in the CSV."
    )

# Define the figures to generate
figures_to_create = [
    {
        "metric": "auc",
        "ylabel": "AUC",
        "title": "Held-out AUC Under Image Degradation",
        "filename": "heldout_auc.png",
        "ylim": (0.45, 1.01),
    },
    {
        "metric": "frr_at_validation_threshold",
        "ylabel": "False rejection rate (FRR)",
        "title": "FRR at the Fixed Validation Threshold",
        "filename": "heldout_frr.png",
        "ylim": (0, 1.02),
    },
]

# Generate both figures
for config in figures_to_create:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, condition in zip(axes, ["brightness", "blur"]):
        rows = sorted(
            [
                row for row in data
                if row["condition"] == condition
            ],
            key=lambda row: row["severity"],
        )

        x = [row["severity"] for row in rows]
        y = [row[config["metric"]] for row in rows]

        ax.plot(x, y, marker="o")
        ax.set_title(condition.capitalize())

        if condition == "brightness":
            ax.set_xlabel("Brightness factor (1.0 = original)")
        else:
            ax.set_xlabel("Gaussian blur radius")

        ax.set_ylabel(config["ylabel"])
        ax.set_ylim(*config["ylim"])
        ax.grid(True, alpha=0.3)

    fig.suptitle(config["title"])
    fig.tight_layout()

    output_path = FIGURES / config["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)

print("\nBoth held-out robustness figures have been generated.")