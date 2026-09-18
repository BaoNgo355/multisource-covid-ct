# Data

This project uses the **RICORD dataset** (Research Imaging Coordinate Resource for COVID-19) from [TCIA](https://tcia.nci.nih.gov/).

## Dataset Overview

| Collection | Description | Subjects | DICOM Series | Total Images |
|------------|-------------|----------|--------------|--------------|
| RICORD-1A | COVID-19 positive CT scans | 110 | 229 | 31,856 |
| RICORD-1B | COVID-19 negative CT scans | 117 | 120 | 21,220 |

## Download

### Option 1: TCIA Data Retriever (Recommended)

1. Download [TCIA Data Retriever](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Data+Retriever) from TCIA website
2. Install and open the application
3. Search for collections:
   - Type `MIDRC-RICORD-1A` in search box → Select collection → Click "Download"
   - Type `MIDRC-RICORD-1B` in search box → Select collection → Click "Download"
4. Choose download location and wait for completion
5. Place the downloaded folders following the structure below

### Option 2: Manual Download

1. Visit [TCIA RICORD-1A](https://tcia.nci.nih.gov/collections/MIDRC-RICORD-1A)
2. Click "Download" → Select "Download Entire Collection"
3. Repeat for [RICORD-1B](https://tcia.nci.nih.gov/collections/MIDRC-RICORD-1B)

### Expected Data Size

- RICORD-1A: ~3GB (229 series, 31,856 DICOM images)
- RICORD-1B: ~2GB (120 series, 21,220 DICOM images)
- Total: ~5GB

## Folder Structure

```
data/
├── dicom/                              # Raw DICOM data (after download)
│   ├── ricord_1a/
│   │   └── manifest-1608266677008/
│   │       ├── metadata.csv            # Series metadata
│   │       └── MIDRC-RICORD-1A/
│   │           └── <subject_id>/       # e.g., MIDRC-RICORD-1A-419639-000082
│   │               └── <study>/
│   │                   └── <series>/
│   │                       └── *.dcm   # DICOM files
│   └── ricord_1b/
│       └── manifest-1612365584013/
│           ├── metadata.csv
│           └── MIDRC-RICORD-1B/
│               └── <subject_id>/
│
├── png/                                # After DICOM→PNG conversion
│   ├── covid/
│   │   └── <subject_id>/
│   │       ├── slice_0000.png
│   │       ├── slice_0001.png
│   │       └── ... (varies per scan)
│   └── non-covid/
│       └── <subject_id>/
│
├── splits/                             # After CSV split creation
│   ├── train_covid.csv                 # 71 scans
│   ├── train_non_covid.csv             # 63 scans
│   ├── validation_covid.csv            # 18 scans
│   └── validation_non_covid.csv        # 16 scans
│
└── preprocessed/                       # After preprocessing
    ├── train/
    │   └── <scan_id>/
    │       ├── slice_0000.png          # 8 KDS-selected slices
    │       ├── slice_0001.png
    │       ├── ...
    │       └── slice_0007.png
    └── val/
        └── <scan_id>/
```

## Pipeline

### Step 1: Convert DICOM to PNG

```bash
python scripts/convert_dicom_to_png.py
```

**What it does:**
- Reads `metadata.csv` to select best axial CT series per subject
- Excludes: SCOUT, COR 3X3, SAG 3X3, bone algorithm series
- Applies lung windowing (W=1500, L=-600)
- Converts DICOM to PNG slices

**Expected output:** 183 scans, 24,056 PNG slices

### Step 2: Create Train/Val Splits

```bash
python scripts/create_csv_splits.py
```

**What it does:**
- Collects scans with ≥5 slices from `data/png/`
- Creates 80/20 stratified train/val split
- Generates CSV files

**Expected output:** 134 train, 34 val scans

### Step 3: Preprocess

```bash
python preprocess.py --raw_dir data/png --output_dir data/preprocessed --csv_dir data/splits
```

**What it does:**
- Reads CSV files for train/val splits
- Applies SSFL lung extraction
- Applies KDS sampling (8 slices per scan)
- Resizes to 256×256

**Expected output:** 134 train + 34 val preprocessed scans

## CSV Format

Each CSV file has columns:

| Column | Type | Description |
|--------|------|-------------|
| `ct_scan_name` | string | Scan folder name (e.g., `MIDRC-RICORD-1A-419639-000082`) |
| `data_centre` | int | Source identifier (always `0` for single-source RICORD) |

**Example:**
```csv
ct_scan_name,data_centre
MIDRC-RICORD-1A-419639-000082,0
MIDRC-RICORD-1A-419639-000361,0
```

## DICOM Series Selection

The `convert_dicom_to_png.py` script uses priority-based series selection:

### Preferred Series (in priority order)

1. ROUTINE CHEST NON-CON
2. NON CON CHEST
3. CHEST WITHOUT CONTRAST
4. ROUTINE CHEST WITH CONTRAST
5. ARTERIAL AXIAL THIN
6. ARTERIAL AXIAL THICK
7. ARTERIALVENOUS AXIAL THIN
8. VENOUS AXIAL THICK
9. THORAX PE ART AXIAL 3X3
10. THORAX ARTERIAL AXIAL 3X3

### Excluded Series

- SCOUT CHEST, PE SCOUT, Scout
- COR 3X3, 3X3 CORONAL, chest coronal
- SAG 3X3, SAG 5X5, chest sagittal
- 0.625mm bone alg, CHEST .625 BONE ALGORITH
- PE Smart Prep Left Atrium, Smart Prep Series
- PE MIP COR 15X5

## Notes

- Only scans with ≥5 slices are included in training
- KDS selects 8 representative slices per scan at 256×256 resolution
- All images are converted with lung windowing (W=1500, L=-600)
- Single-source mode (γ=0.0) means all scans are assigned to source 0
