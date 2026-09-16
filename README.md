# Robust Multi-Source COVID-19 Detection in CT Images

<p align="center">
  Adapted for RICORD Single-Source Training
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1-orange" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="#getting-started">[Getting Started]</a> •
  <a href="#results">[Results]</a> •
  <a href="#method">[Method]</a> •
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
| GPU | NVIDIA A100 | Consumer GPU (4GB VRAM) |

---

## Results

### Single-Source RICORD Training

| Metric | Value |
|--------|-------|
| **F1 Score** | 0.7879 |
| **AUC-ROC** | 0.8056 |
| **Competition Score** | **0.7939** |
| **Accuracy** | 79.4% |
| **Sensitivity** | 72.2% |
| **Specificity** | 87.5% |

```
Confusion Matrix
              Pred    Non-COVID    COVID
Actual
Non-COVID           14 (TN)       2 (FP)
COVID                5 (FN)      13 (TP)
```

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-COVID | 0.74 | 0.88 | 0.80 | 16 |
| COVID | 0.87 | 0.72 | 0.79 | 18 |

---

## Getting Started

### 1. Environment

```bash
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scipy opencv-python pydicom pandas tqdm
```

Tested with Python 3.10+, PyTorch 2.1+, CUDA 12.1 on a single NVIDIA GPU (4GB VRAM).

### 2. Data Download

Download DICOM data from [TCIA](https://tcia.nci.nih.gov/) using the **TCIA Data Retriever** tool:

- **RICORD-1A** (COVID-positive): 110 series → 89 valid scans
- **RICORD-1B** (COVID-negative): 117 series → 79 valid scans

Place manifest files and DICOM data in:
```
data/dicom/
├── ricord_1a/
│   └── manifest-1608266677008/
│       ├── metadata.csv
│       └── MIDRC-RICORD-1A/...
└── ricord_1b/
    └── manifest-1612365584013/
        ├── metadata.csv
        └── MIDRC-RICORD-1B/...
```

See [`data/README.md`](data/README.md) for detailed instructions.

### 3. DICOM → PNG Conversion

```bash
python scripts/convert_dicom_to_png.py
```

This script:
- Reads TCIA metadata to select best axial CT series per subject
- Applies lung windowing (W=1500, L=-600)
- Converts DICOM to PNG slices

Output: `data/png/covid/` and `data/png/non-covid/`

### 4. Create Train/Val Splits

```bash
python scripts/create_csv_splits.py
```

Generates 80/20 stratified splits as CSV files:
```
data/splits/
├── train_covid.csv
├── train_non_covid.csv
├── validation_covid.csv
└── validation_non_covid.csv
```

### 5. Preprocessing

```bash
python preprocess.py \
    --raw_dir data/png \
    --output_dir data/preprocessed \
    --csv_dir data/splits
```

Applies SSFL lung extraction and KDS sampling (8 slices/scan, 256×256).

### 6. Training

```bash
python train.py \
    --data_dir data/preprocessed \
    --csv_dir data/splits \
    --gamma 0.0 \
    --epochs 20 \
    --batch_size 2
```

- `gamma=0.0`: Single-source mode (no source identification loss)
- `batch_size=2`: Adjust based on GPU VRAM (2 for 4GB, 10 for 16GB+)
- Best model saved to `checkpoints/best_gamma0.0.pth`

### 7. Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_gamma0.0.pth \
    --data_dir data/preprocessed \
    --csv_dir data/splits
```

Outputs accuracy, F1, AUC-ROC, sensitivity, specificity, and confusion matrix.

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
├── requirements.txt
├── LICENSE
└── README.md
```

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
