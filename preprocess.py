"""
Run SSFL lung extraction + KDS preprocessing on raw CT scans.

Usage:
    python preprocess.py --raw_dir data/png --output_dir data/preprocessed --csv_dir data/splits
"""

import os
import argparse
import pandas as pd
from src.preprocessing import is_valid_image, preprocess_scans


def collect_scans_from_dir(label_dir, label, label_name, scan_ids=None):
    """Collect scans from a directory."""
    scans = []
    if not os.path.exists(label_dir):
        return scans
    for scan_name in sorted(os.listdir(label_dir)):
        scan_path = os.path.join(label_dir, scan_name)
        if not os.path.isdir(scan_path) or scan_name.startswith('_'):
            continue
        # If scan_ids provided, only include those in the list
        if scan_ids is not None and scan_name not in scan_ids:
            continue
        slices = sorted([
            os.path.join(scan_path, f)
            for f in os.listdir(scan_path) if is_valid_image(f)
        ])
        if len(slices) >= 5:
            scans.append({
                'scan_id': scan_name,
                'slices': slices,
                'label': label,
                'label_name': label_name,
            })
    return scans


def load_scan_ids_from_csv(csv_path):
    """Load scan names from CSV file."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return set(df['ct_scan_name'].values)


def main(args):
    print("=" * 60)
    print("Preprocessing: Lung Extraction + KDS Sampling")
    print("=" * 60)

    # Load train/val splits from CSV
    train_covid_ids = load_scan_ids_from_csv(os.path.join(args.csv_dir, 'train_covid.csv'))
    train_noncovid_ids = load_scan_ids_from_csv(os.path.join(args.csv_dir, 'train_non_covid.csv'))
    val_covid_ids = load_scan_ids_from_csv(os.path.join(args.csv_dir, 'validation_covid.csv'))
    val_noncovid_ids = load_scan_ids_from_csv(os.path.join(args.csv_dir, 'validation_non_covid.csv'))

    print(f"Train COVID: {len(train_covid_ids) if train_covid_ids else 0} scans")
    print(f"Train Non-COVID: {len(train_noncovid_ids) if train_noncovid_ids else 0} scans")
    print(f"Val COVID: {len(val_covid_ids) if val_covid_ids else 0} scans")
    print(f"Val Non-COVID: {len(val_noncovid_ids) if val_noncovid_ids else 0} scans")

    # Collect train scans
    train_scans = []
    if train_covid_ids:
        train_scans.extend(collect_scans_from_dir(
            os.path.join(args.raw_dir, 'covid'), 1, 'covid', train_covid_ids
        ))
    if train_noncovid_ids:
        train_scans.extend(collect_scans_from_dir(
            os.path.join(args.raw_dir, 'non-covid'), 0, 'non-covid', train_noncovid_ids
        ))

    # Collect val scans
    val_scans = []
    if val_covid_ids:
        val_scans.extend(collect_scans_from_dir(
            os.path.join(args.raw_dir, 'covid'), 1, 'covid', val_covid_ids
        ))
    if val_noncovid_ids:
        val_scans.extend(collect_scans_from_dir(
            os.path.join(args.raw_dir, 'non-covid'), 0, 'non-covid', val_noncovid_ids
        ))

    print(f"\nTrain total: {len(train_scans)} scans")
    print(f"Val total: {len(val_scans)} scans")

    # Run preprocessing
    preprocess_scans(train_scans, os.path.join(args.output_dir, 'train'), 'train')
    preprocess_scans(val_scans, os.path.join(args.output_dir, 'val'), 'val')

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw CT scans')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to raw PNG data (contains covid/ and non-covid/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save preprocessed scans')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Path to directory with train/val CSV files')
    args = parser.parse_args()
    main(args)
