# Data

This project uses the **RICORD dataset** (Research Imaging Coordinate Resource for COVID-19) from [TCIA](https://tcia.nci.nih.gov/).

## Dataset

| Collection | Description | Subjects | Scans |
|------------|-------------|----------|-------|
| RICORD-1A | COVID-19 positive CT scans | 110 | 89 |
| RICORD-1B | COVID-19 negative CT scans | 117 | 79 |

## Download

### Option 1: TCIA Data Retriever (Recommended)

1. Download [TCIA Data Retriever](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Data+Retriever)
2. Search for collections:
   - **MIDRC-RICORD-1A** (COVID-positive)
   - **MIDRC-RICORD-1B** (COVID-negative)
3. Download manifests and DICOM data
4. Place files following the structure below

### Option 2: Manual Download

Visit [TCIA RICORD](https://tcia.nci.nih.gov/collections?search=RICORD) and download DICOM files manually.

## Folder Structure

```
data/
├── dicom/
│   ├── ricord_1a/
│   │   ├── manifest-1608266677008/
│   │   │   ├── metadata.csv
│   │   │   └── MIDRC-RICORD-1A/
│   │   │       └── <subject_id>/
│   │   │           └── <study>/
│   │   │               └── <series>/
│   │   │                   └── *.dcm
│   └── ricord_1b/
│       └── manifest-1612365584013/
│           ├── metadata.csv
│           └── MIDRC-RICORD-1B/
│               └── <subject_id>/
├── png/                          # After DICOM→PNG conversion
│   ├── covid/
│   │   └── <subject_id>/
│   │       └── slice_0000.png
│   └── non-covid/
│       └── <subject_id>/
├── splits/                       # After CSV split creation
│   ├── train_covid.csv
│   ├── train_non_covid.csv
│   ├── validation_covid.csv
│   └── validation_non_covid.csv
└── preprocessed/                 # After preprocessing
    ├── train/
    │   └── <scan_id>/
    │       ├── slice_*.png       # 8 KDS-selected slices
    │       └── scan_info.json
    └── val/
        └── <scan_id>/
```

## Pipeline

```bash
# Step 1: Convert DICOM to PNG with lung windowing
python scripts/convert_dicom_to_png.py

# Step 2: Create train/val CSV splits (80/20)
python scripts/create_csv_splits.py

# Step 3: Preprocess (SSFL lung extraction + KDS sampling)
python preprocess.py --raw_dir data/png --output_dir data/preprocessed --csv_dir data/splits
```

## CSV Format

Each CSV file has columns:
- `ct_scan_name`: Name of the scan folder
- `data_centre`: Source identifier (0 for single-source RICORD)

Example:
```csv
ct_scan_name,data_centre
MIDRC-RICORD-1A-00001,0
MIDRC-RICORD-1A-00002,0
```

## Notes

- Exclude SCOUT, coronal, sagittal, and bone algorithm series during DICOM selection
- Only scans with ≥5 slices are included
- KDS selects 8 representative slices per scan at 256×256 resolution
