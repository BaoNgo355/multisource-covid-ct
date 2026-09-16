"""
Convert DICOM files to PNG with lung windowing (W=-600, L=1500).

Reads metadata.csv from TCIA Data Retriever output, selects the best
axial CT series per subject, and converts to PNG.

Usage:
    python scripts/convert_dicom_to_png.py
"""

import os
import csv
import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DICOM_DIR = BASE_DIR / "data" / "dicom"
PNG_DIR = BASE_DIR / "data" / "png"

# Lung windowing parameters
WINDOW_WIDTH = 1500
WINDOW_LEVEL = -600

# Series to exclude (non-axial or non-diagnostic)
EXCLUDE_SERIES = {
    "SCOUT CHEST", "SCOUT SUPINE", "Scout", "PE SCOUT",
    "Scout 4CM ABOVE STERNAL NOTCH",
    "COR 3X3", "3X3 CORONAL", "chest cor",
    "SAG 3X3", "SAG 5X5", "chest sag",
    "PE MIP COR 15X5", "PE Smart Prep Left Atrium",
    "Smart Prep Series",
    "0.625mm bone alg", "0.625mm bone alg chest",
    "CHEST .625 BONE ALGORITH",
    "2.5mm IODINEWATER",
    "CTA  SP ON ASCENDING AORTA",
    "NA",
}

# Preferred axial series (in priority order)
PREFERRED_SERIES = [
    "ROUTINE CHEST NON-CON",
    "NON CON CHEST",
    "CHEST WITHOUT CONTRAST",
    "ROUTINE CHEST WITH CONTRAST",
    "CHEST WITHOUT CONTRAST",
    "chest",
    "ARTERIAL AXIAL THIN",
    "ARTERIAL AXIAL THICK",
    "ARTERIALVENOUS AXIAL THIN",
    "VENOUS AXIAL THICK",
    "THORAX PE ART AXIAL 3X3",
    "THORAX ARTERIAL AXIAL 3X3",
    "THORAX LD STD AXIAL 3X3",
    "THORAX ARTVEN AXIAL 3X3",
    "TAP STD AXIAL 3X3",
    "0.625mm Detail alg",
    "1.25 60 KEV",
    "SUPINE CHEST RECON 12",
    "1.25mm CHEST Stnd Alg",
]


def apply_lung_window(pixel_array, window_width=WINDOW_WIDTH, window_level=WINDOW_LEVEL):
    """Apply lung windowing to DICOM pixel array."""
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2
    img = np.clip(pixel_array, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img


def read_metadata(csv_path):
    """Read TCIA metadata.csv and return list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def select_best_series(rows):
    """Select best axial CT series for each subject.

    Returns dict: subject_id -> (series_description, file_location, num_images)
    """
    subjects = {}
    for row in rows:
        subj = row["Subject ID"]
        desc = row["Series Description"]
        loc = row["File Location"]
        num = int(row["Number of Images"])

        if subj not in subjects:
            subjects[subj] = []

        subjects[subj].append({
            "description": desc,
            "location": loc,
            "num_images": num,
        })

    # Select best series per subject
    best = {}
    for subj, series_list in subjects.items():
        # Filter out excluded series
        candidates = [s for s in series_list if s["description"] not in EXCLUDE_SERIES]
        if not candidates:
            # If all excluded, try to find any with enough slices
            candidates = [s for s in series_list if s["num_images"] >= 50]
        if not candidates:
            continue

        # Sort by preferred order, then by num_images
        def sort_key(s):
            try:
                pref_idx = PREFERRED_SERIES.index(s["description"])
            except ValueError:
                pref_idx = len(PREFERRED_SERIES)
            return (pref_idx, -s["num_images"])

        candidates.sort(key=sort_key)
        best[subj] = candidates[0]

    return best


def convert_subject_to_png(subject_id, series_info, output_dir, label, manifest_dir):
    """Convert all DICOM files in a series to PNG."""
    loc = series_info["location"]
    # Strip leading "./" or ".\" prefix from TCIA paths
    loc = loc.lstrip(".").lstrip("\\").lstrip("/")
    base_path = (manifest_dir / loc).resolve()

    if not base_path.exists():
        print(f"  [WARN] Path not found: {base_path}")
        return 0

    # Find all .dcm files
    try:
        all_files = os.listdir(base_path)
    except OSError as e:
        print(f"  [ERROR] Cannot list {base_path}: {e}")
        return 0
    dcm_files = sorted([f for f in all_files if f.lower().endswith(".dcm")])

    if not dcm_files:
        return 0

    # Create output directory
    scan_dir = output_dir / label / subject_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for dcm_file in dcm_files:
        try:
            dcm_path = base_path / dcm_file
            ds = pydicom.dcmread(str(dcm_path))
            pixel_array = ds.pixel_array.astype(np.float64)

            # Apply rescale if present
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

            # Apply lung windowing
            img = apply_lung_window(pixel_array)

            # Save as PNG
            png_name = f"slice_{count:04d}.png"
            cv2.imwrite(str(scan_dir / png_name), img)
            count += 1
        except Exception as e:
            if count == 0:
                print(f"  [ERROR] {subject_id}: {e}")
            continue

    return count


def process_dataset(collection_name, label, metadata_csv, manifest_dir):
    """Process one RICORD collection."""
    print(f"\nProcessing {collection_name} ({label})...")

    rows = read_metadata(metadata_csv)
    print(f"  Total series in metadata: {len(rows)}")

    best_series = select_best_series(rows)
    print(f"  Subjects with axial series: {len(best_series)}")

    output_dir = PNG_DIR
    total_slices = 0
    processed = 0

    for subj_id, info in tqdm(best_series.items(), desc=f"  Converting {collection_name}"):
        n = convert_subject_to_png(subj_id, info, output_dir, label, manifest_dir)
        if n > 0:
            total_slices += n
            processed += 1

    print(f"  Processed: {processed} scans, {total_slices} slices total")
    return processed, total_slices


def main():
    print("=" * 60)
    print("DICOM -> PNG Conversion with Lung Windowing")
    print(f"Window: W={WINDOW_WIDTH}, L={WINDOW_LEVEL}")
    print("=" * 60)

    # Ensure output directories exist
    (PNG_DIR / "covid").mkdir(parents=True, exist_ok=True)
    (PNG_DIR / "non-covid").mkdir(parents=True, exist_ok=True)

    # Process RICORD-1A (COVID-positive)
    manifest_1a = DICOM_DIR / "ricord_1a" / "manifest-1608266677008"
    metadata_1a = manifest_1a / "metadata.csv"
    n_scans_1a, n_slices_1a = process_dataset(
        "RICORD-1A", "covid", metadata_1a, manifest_1a
    )

    # Process RICORD-1B (COVID-negative)
    manifest_1b = DICOM_DIR / "ricord_1b" / "manifest-1612365584013"
    metadata_1b = manifest_1b / "metadata.csv"
    n_scans_1b, n_slices_1b = process_dataset(
        "RICORD-1B", "non-covid", metadata_1b, manifest_1b
    )

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"  COVID (RICORD-1A):     {n_scans_1a} scans, {n_slices_1a} slices")
    print(f"  Non-COVID (RICORD-1B): {n_scans_1b} scans, {n_slices_1b} slices")
    print(f"  Total:                 {n_scans_1a + n_scans_1b} scans, {n_slices_1a + n_slices_1b} slices")
    print(f"\nOutput directory: {PNG_DIR}")


if __name__ == "__main__":
    main()
