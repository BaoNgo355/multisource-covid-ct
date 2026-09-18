# Robust Multi-Source COVID-19 Detection in CT Images

<p align="center">
  Adapted for RICORD Single-Source Training
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1-orange" alt="PyTorch"></a>
  <a href="REPORT.md"><img src="https://img.shields.io/badge/Report-MD-blue" alt="Report"></a>
</p>

<p align="center">
  <a href="#getting-started">[Getting Started]</a> •
  <a href="#results">[Results]</a> •
  <a href="#report">[Full Report]</a> •
  <a href="#acknowledgements">[Acknowledgements]</a>
</p>

> Based on the work by **Asmita Yuki Pritha**, **Jason Xu**, **Daniel Ding**, **Justin Li**, **Aryana Hou**, Xin Wang, Shu Hu†
>
> M2 Lab, Purdue University
>
> *Paper accepted at 3rd Workshop on New Trends in AI-Generated Media and Security (AIMS) @ CVPR 2026*

---

## Overview

This project adapts the multi-task COVID-19 CT detection framework from [Purdue-M2/multisource-covid-ct](https://github.com/Purdue-M2/multisource-covid-ct) for single-source training on the **RICORD dataset** (COVID-19-positive and COVID-19-negative CT scans from TCIA).

### Changes from Original

| Aspect | Original (PHAROS) | This Fork (RICORD) |
|--------|-------------------|---------------------|
| Dataset | PHAROS Multi-Source (4 hospitals) | RICORD-1A + RICORD-1B (single source) |
| Sources | 4 sources, logit-adjusted CE | 1 source, BCE only (γ=0.0) |
| Data format | Pre-processed PNG | DICOM → PNG conversion |
| Preprocessing | Fixed folder structure | CSV-based train/val splits |
| GPU | NVIDIA A100 (40GB) | Consumer GPU (4GB VRAM) |

---

## Results

| Metric | Value | Clinical Requirement |
|--------|-------|---------------------|
| **F1 Score** | 0.7879 | >0.85 |
| **AUC-ROC** | 0.8056 | >0.85 |
| **Competition Score** | **0.7939** | - |
| **Accuracy** | 79.4% | - |
| **Sensitivity** | 72.2% | >90% |
| **Specificity** | 87.5% | >85% ✅ |

```
Confusion Matrix
              Pred    Non-COVID    COVID
Actual
Non-COVID           14 (TN)       2 (FP)
COVID                5 (FN)      13 (TP)
```

> See [REPORT.md](REPORT.md) for detailed analysis of all metrics.

---

## Getting Started

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.10+ |
| GPU VRAM | 4GB | 8GB+ |
| RAM | 8GB | 16GB |
| Disk space | 20GB | 50GB |
| OS | Windows/Linux | Windows/Linux |

### Step 1: Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install timm albumentations scipy opencv-python pydicom pandas tqdm
```

**Verify installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce GTX XXXX  # your GPU name
```

**Time**: ~5-10 minutes

### Step 2: Download DICOM Data

**Option A: TCIA Data Retriever (Recommended)**

1. Download [TCIA Data Retriever](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Data+Retriever)
2. Install and open the application
3. Search for collections:
   - Search `MIDRC-RICORD-1A` → Select → Download (all series)
   - Search `MIDRC-RICORD-1B` → Select → Download (all series)
4. Place downloaded data following the structure below

**Option B: Manual Download**

1. Visit [TCIA RICORD-1A](https://tcia.nci.nih.gov/collections/MIDRC-RICORD-1A)
2. Click "Download" → Select "Download Entire Collection"
3. Repeat for [RICORD-1B](https://tcia.nci.nih.gov/collections/MIDRC-RICORD-1B)

**Expected folder structure:**
```
data/dicom/
├── ricord_1a/
│   └── manifest-1608266677008/
│       ├── metadata.csv              # Series metadata
│       └── MIDRC-RICORD-1A/
│           ├── MIDRC-RICORD-1A-419639-000082/
│           │   └── 08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/
│           │       └── 2.000000-ROUTINE CHEST NON-CON-97100/
│           │           └── *.dcm
│           └── ... (110 subjects)
└── ricord_1b/
    └── manifest-1612365584013/
        ├── metadata.csv
        └── MIDRC-RICORD-1B/
            └── ... (117 subjects)
```

**Expected data size:**
- RICORD-1A: ~3GB (229 series, 31,856 DICOM images)
- RICORD-1B: ~2GB (120 series, 21,220 DICOM images)
- Total: ~5GB

**Time**: ~30-60 minutes (depending on internet speed)

### Step 3: Convert DICOM to PNG

```bash
python scripts/convert_dicom_to_png.py
```

**What it does:**
- Reads TCIA metadata to select best axial CT series per subject
- Excludes: SCOUT, coronal (COR 3X3), sagittal (SAG 3X3), bone algorithm series
- Applies lung windowing (Window Width=1500, Window Level=-600)
- Converts DICOM to PNG slices

**Expected output:**
```
Processing RICORD-1A (covid)...
  Total series in metadata: 229
  Subjects with axial series: 92
  Converting RICORD-1A: 92 scans, 11797 slices total

Processing RICORD-1B (non-covid)...
  Total series in metadata: 120
  Subjects with axial series: 91
  Converting RICORD-1B: 91 scans, 12259 slices total

CONVERSION SUMMARY
  COVID (RICORD-1A):     92 scans, 11797 slices
  Non-COVID (RICORD-1B): 91 scans, 12259 slices
  Total:                 183 scans, 24056 slices
```

**Output folders:**
```
data/png/
├── covid/
│   ├── MIDRC-RICORD-1A-419639-000082/
│   │   ├── slice_0000.png
│   │   ├── slice_0001.png
│   │   └── ...
│   └── ... (92 scans)
└── non-covid/
    └── ... (91 scans)
```

**Time**: ~5-10 minutes

### Step 4: Create Train/Val Splits

```bash
python scripts/create_csv_splits.py
```

**What it does:**
- Collects all valid scans (≥5 slices) from `data/png/`
- Creates 80/20 stratified train/val split
- Generates CSV files with scan names and source labels

**Expected output:**
```
Creating Train/Val CSV Splits
============================================================
COVID scans: 92
Non-COVID scans: 91

Train COVID: 71 | Val COVID: 18
Train Non-COVID: 63 | Val Non-COVID: 16

CSV files saved to: data/splits
Files:
  - train_covid.csv
  - train_non_covid.csv
  - validation_covid.csv
  - validation_non_covid.csv
```

**CSV format:**
```csv
ct_scan_name,data_centre
MIDRC-RICORD-1A-419639-000082,0
MIDRC-RICORD-1A-419639-000361,0
```

**Time**: ~1 minute

### Step 5: Preprocess (SSFL + KDS)

```bash
python preprocess.py --raw_dir data/png --output_dir data/preprocessed --csv_dir data/splits
```

**What it does:**
- Reads CSV files to determine train/val splits
- Applies SSFL lung extraction (spatial filtering, binarization, morphological closing)
- Applies KDS sampling (selects 8 representative slices per scan)
- Resizes all slices to 256×256

**Expected output:**
```
Preprocessing: Lung Extraction + KDS Sampling
============================================================
Train COVID: 71 scans
Train Non-COVID: 63 scans
Val COVID: 18 scans
Val Non-COVID: 16 scans

Train total: 134 scans
Val total: 34 scans
```

**Output folder:**
```
data/preprocessed/
├── train/
│   ├── MIDRC-RICORD-1A-419639-000082/
│   │   ├── slice_0000.png  # 8 KDS-selected slices
│   │   ├── slice_0001.png
│   │   ├── ...
│   │   └── slice_0007.png
│   └── ... (134 scans)
└── val/
    └── ... (34 scans)
```

**Time**: ~10-15 minutes

### Step 6: Train Model

```bash
python train.py --data_dir data/preprocessed --csv_dir data/splits --gamma 0.0 --epochs 20 --batch_size 2
```

**Parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--gamma 0.0` | Single-source mode | No source identification loss |
| `--epochs 20` | Training epochs | Best checkpoint at epoch 14 |
| `--batch_size 2` | Batch size | Adjust for GPU: 2 (4GB), 10 (16GB+) |
| `--lr 1e-4` | Learning rate | Default, Adam optimizer |

**Expected output:**
```
Device: cuda
Train: 134 | Val: 34
Training with gamma=0.0 for 20 epochs

Epoch 1/20
  Train  loss=0.6500  F1=0.6500
  Val    loss=0.7500  F1=0.5185  AUC=0.7569
...
Epoch 14/20
  Train  loss=0.5762  F1=0.7273
  Val    loss=0.6338  F1=0.7333  AUC=0.7674  ← Best checkpoint saved

FINAL RESULTS
F1=0.7879  AUC=0.8056  Final Score=0.7939
Checkpoint saved to ./checkpoints/best_gamma0.0.pth
```

**Time**: ~3.5 hours (4GB GPU, 20 epochs)

### Step 7: Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_gamma0.0.pth --data_dir data/preprocessed --csv_dir data/splits
```

**Expected output:**
```
VALIDATION RESULTS
============================================================
  Accuracy:    0.7941
  F1 Score:    0.7879
  AUC-ROC:     0.8056
  Sensitivity: 0.7222
  Specificity: 0.8750

Confusion Matrix
  TN=14  FP=2
  FN=5  TP=13

              precision    recall  f1-score   support
   Non-COVID       0.74      0.88      0.80        16
       COVID       0.87      0.72      0.79        18
    accuracy                           0.79        34

COMPETITION SCORE = 0.7939
```

**Time**: ~1 minute

---

## Quick Start (All Steps)

```bash
# Setup
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scipy opencv-python pydicom pandas tqdm

# Data (after downloading DICOM from TCIA)
python scripts/convert_dicom_to_png.py
python scripts/create_csv_splits.py
python preprocess.py --raw_dir data/png --output_dir data/preprocessed --csv_dir data/splits

# Train
python train.py --data_dir data/preprocessed --csv_dir data/splits --gamma 0.0 --epochs 20 --batch_size 2

# Evaluate
python evaluate.py --checkpoint checkpoints/best_gamma0.0.pth --data_dir data/preprocessed --csv_dir data/splits
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--batch_size` to 1 |
| `ModuleNotFoundError: No module named 'timm'` | Run `pip install timm` |
| `FileNotFoundError: data/dicom/...` | Check DICOM folder structure matches data/README.md |
| `RuntimeError: CUDA error: device-side assert triggered` | Check GPU supports CUDA 12.1 |
| Training too slow | Use GPU with more VRAM, increase batch_size |
| Low F1/Sensitivity | Try more epochs, stronger augmentation, or more data |

---

## Project Structure

```
├── configurations/
│   └── default.yaml              # Hyperparameters and augmentation config
├── data/
│   ├── README.md                 # Dataset download instructions
│   ├── dicom/                    # Raw DICOM data (not tracked)
│   ├── png/                      # Converted PNG images (not tracked)
│   ├── splits/                   # Train/val CSV splits
│   └── preprocessed/             # Preprocessed scans (not tracked)
├── scripts/
│   ├── convert_dicom_to_png.py   # DICOM → PNG with lung windowing
│   ├── create_csv_splits.py      # Create train/val CSV splits
│   ├── sweep_gamma.sh            # γ sweep script (original)
│   └── visualize_results.py      # Generate paper figures (original)
├── src/
│   ├── __init__.py
│   ├── model.py                  # Multi-task EfficientNet-B7
│   ├── losses.py                 # BCE + Logit-Adjusted CE
│   ├── dataset.py                # CT dataset class and augmentations
│   ├── preprocessing.py          # Lung extraction + KDS sampling
│   └── engine.py                 # Training and evaluation loops
├── checkpoints/                  # Model weights (not tracked)
├── train.py                      # Main training script
├── preprocess.py                 # Data preprocessing script
├── evaluate.py                   # Standalone evaluation script
├── REPORT.md                     # Detailed project report
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Report

See [REPORT.md](REPORT.md) for detailed analysis including:
- Dataset statistics (RICORD-1A and RICORD-1B)
- Evaluation metrics explanation
- Training analysis
- Error analysis
- Comparison with original
- Potential improvements

---

## Method

1. **Preprocessing** — SSFL lung extraction isolates the lung region via spatial filtering, binarization, and morphological closing. KDS fits a Gaussian KDE over slice-level lung areas and selects 8 representative slices per scan.

2. **Architecture** — EfficientNet-B7 processes each slice independently, producing 8 feature vectors of dimension 2560. Element-wise mean pooling aggregates them into a single scan-level representation, which feeds two heads: a binary COVID-19 classifier and a source identifier.

3. **Loss** — The COVID head uses BCE. In single-source mode (γ=0.0), only the COVID loss is used.

---

## Acknowledgements

This work builds upon the multi-source COVID-19 detection framework developed by the M2 Lab at Purdue University, supported by the U.S. National Science Foundation (NSF) under grant IIS-2434967, and the National Artificial Intelligence Research Resource (NAIRR) Pilot and TACC Lonestar6.

Dataset: [RICORD](https://ricord.org/) from [TCIA](https://tcia.nci.nih.gov/).

## License

This project is released under the [MIT License](LICENSE).
