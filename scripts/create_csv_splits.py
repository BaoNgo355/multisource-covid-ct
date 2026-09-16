"""
Create train/val CSV splits for RICORD dataset.

Generates:
  - train_covid.csv, train_non_covid.csv
  - validation_covid.csv, validation_non_covid.csv

Each CSV has columns: ct_scan_name, data_centre

Usage:
    python scripts/create_csv_splits.py
"""

import os
import csv
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PNG_DIR = BASE_DIR / "data" / "png"
CSV_DIR = BASE_DIR / "data" / "splits"

TRAIN_RATIO = 0.8
SEED = 42


def collect_scans(label_dir):
    """Collect scan folders from a label directory."""
    scans = []
    if not label_dir.exists():
        return scans
    for scan_name in sorted(os.listdir(label_dir)):
        scan_path = label_dir / scan_name
        if scan_path.is_dir():
            # Count PNG slices
            pngs = [f for f in os.listdir(scan_path) if f.endswith(".png")]
            if len(pngs) >= 5:  # Minimum 5 slices
                scans.append(scan_name)
    return scans


def create_csv(rows, csv_path):
    """Write CSV file."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ct_scan_name", "data_centre"])
        for row in rows:
            writer.writerow(row)


def main():
    print("=" * 60)
    print("Creating Train/Val CSV Splits")
    print("=" * 60)

    # Create output directory
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Collect scans
    covid_scans = collect_scans(PNG_DIR / "covid")
    noncovid_scans = collect_scans(PNG_DIR / "non-covid")

    print(f"COVID scans: {len(covid_scans)}")
    print(f"Non-COVID scans: {len(noncovid_scans)}")

    # Set random seed for reproducibility
    random.seed(SEED)

    # Split COVID scans
    covid_shuffled = covid_scans.copy()
    random.shuffle(covid_shuffled)
    covid_split = int(len(covid_shuffled) * TRAIN_RATIO)
    covid_train = covid_shuffled[:covid_split]
    covid_val = covid_shuffled[covid_split:]

    # Split Non-COVID scans
    noncovid_shuffled = noncovid_scans.copy()
    random.shuffle(noncovid_shuffled)
    noncovid_split = int(len(noncovid_shuffled) * TRAIN_RATIO)
    noncovid_train = noncovid_shuffled[:noncovid_split]
    noncovid_val = noncovid_shuffled[noncovid_split:]

    print(f"\nTrain COVID: {len(covid_train)} | Val COVID: {len(covid_val)}")
    print(f"Train Non-COVID: {len(noncovid_train)} | Val Non-COVID: {len(noncovid_val)}")

    # Create CSV files (all with data_centre=0 for single source)
    create_csv(
        [(name, 0) for name in covid_train],
        CSV_DIR / "train_covid.csv"
    )
    create_csv(
        [(name, 0) for name in noncovid_train],
        CSV_DIR / "train_non_covid.csv"
    )
    create_csv(
        [(name, 0) for name in covid_val],
        CSV_DIR / "validation_covid.csv"
    )
    create_csv(
        [(name, 0) for name in noncovid_val],
        CSV_DIR / "validation_non_covid.csv"
    )

    print(f"\nCSV files saved to: {CSV_DIR}")
    print("Files:")
    for f in sorted(CSV_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
