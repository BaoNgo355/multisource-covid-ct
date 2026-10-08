# Data

This project uses the **SARS-CoV-2 CT-scan dataset** from [Kaggle](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset).

## Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | SARS-CoV-2 CT-scan dataset |
| **Authors** | Soares, Angelov, Biaso, Froes, Abe (2020) |
| **Source** | Hospitals in São Paulo, Brazil |
| **Images** | 2,482 PNG (1,252 COVID + 1,230 Non-COVID) |
| **Size** | ~242 MB |
| **License** | CC BY-NC-SA 4.0 |
| **Paper** | [medRxiv 10.1101/2020.04.24.20078584](https://doi.org/10.1101/2020.04.24.20078584) |
| **Baseline (authors)** | xDNN — F1 = 97.31% |

Each image is a single 2D CT slice of the chest (grayscale PNG, sizes vary from ~146 to ~490 px).
There are **no patient IDs, no scan grouping and no source/hospital labels** — each image is an independent sample.

## Download

### Option A: Kaggle CLI (Recommended)

```bash
pip install kaggle

# 1. Create API token: https://www.kaggle.com/settings -> API -> Create New Token
# 2. Place kaggle.json in ~/.kaggle/ (Linux/macOS) or C:\Users\<user>\.kaggle\ (Windows)

kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p data --unzip
```

**Time**: ~5 minutes (242 MB).

### Option B: Manual Download

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
2. Click **Download** (uncompressed .zip)
3. Extract so that the two class folders sit somewhere under `data/`:

```
data/
├── COVID/            # 1,252 CT images (COVID positive)
└── non-COVID/        # 1,230 CT images (COVID negative)
```

> The notebook **auto-detects** the two class folders at any nesting depth — common Kaggle
> folder names (`CT_COVID` / `CT_NonCOVID`) also work. It ignores `preprocessed_*`, `raw_*`
> and `processed_lung_ct_scan` folders so re-runs are safe.

## Expected Structure After Running the Notebook

```
data/
├── README.md
├── COVID/                        # raw (untracked by git)
├── non-COVID/                    # raw (untracked by git)
└── preprocessed_sarscov2/        # generated (untracked by git)
    ├── train/
    │   ├── covid/                # 1,001 images (SSFL lung-extracted, 256×256)
    │   └── non-covid/            #   983 images
    └── val/
        ├── covid/                #   251 images
        └── non-covid/            #   246 images
```

## Split (Stratified 80/20, seed = 42)

| Split | COVID | Non-COVID | Total |
|-------|-------|-----------|-------|
| Train | 1,001 | 983 | **1,984** |
| Validation | 251 | 246 | **497** |
| **Total** | 1,252 | 1,229 | **2,481** |

> Note: 2,481 images on disk vs. 2,482 published — one image from the original release is
> missing locally; the imbalance (50.4% / 49.6%) is unaffected.

## Preprocessing Pipeline

1. **Stratified split first** (80/20, seed 42) — split before preprocessing so no validation
   image influences any transform fitted on training data.
2. **SSFL lung extraction** (kept from the original code): spatial filtering → Otsu
   binarization → contour detection → morphological closing → border removal → resize 256×256.
3. **KDS slice sampling removed** — the original pipeline grouped 8 slices per scan; here each
   file is an independent 2D image (1 image = 1 sample).
4. Toggle `USE_LUNG_EXTRACT = False` in the notebook to train on raw images (ablation).

## Data Quality Checks (performed on this copy)

| Check | Result |
|-------|--------|
| Corrupted images (sample of 80) | 0 |
| Exact duplicates (same MD5) | 2 images (0.08%) |
| Duplicates crossing classes | 0 |
| Duplicate pair crossing train/val | 1 pair (minor leakage, see REPORT.md §9) |

## Legacy / Previous Experiments

Earlier experiments of this repo used other datasets — kept for reference in git history:

- **PHAROS Multi-Source** (original paper): 4 hospital sources, 1,222 scans / 9,776 images,
  scan-level CSV splits, source labels for multi-task training.
- **MIDRC-RICORD** (TCIA): DICOM series converted to PNG via
  `scripts/convert_dicom_to_png.py`, splits via `scripts/create_csv_splits.py`
  (`data/dicom/`, `data/png/` ignored by git).

The current notebook (`MultiSource_COVID_CT.ipynb`) does **not** depend on those scripts.
