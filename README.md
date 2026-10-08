# Robust Multi-Source COVID-19 Detection in CT Images

<p align="center">
  Adapted for SARS-CoV-2 CT-scan Binary Classification (PyCharm Notebook)
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5-orange" alt="PyTorch"></a>
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

This project adapts the COVID-19 CT detection framework from [Purdue-M2/multisource-covid-ct](https://github.com/Purdue-M2/multisource-covid-ct) to train **EfficientNet-B7** on the **[SARS-CoV-2 CT-scan dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)** (2,482 CT images, hospitals in São Paulo, Brazil) as a **binary COVID / Non-COVID classifier**, fully runnable **locally in PyCharm** from a Jupyter notebook.

### Changes from Original

| Aspect | Original (PHAROS) | This Fork (SARS-CoV-2 CT-scan) |
|--------|-------------------|--------------------------------|
| Dataset | PHAROS Multi-Source (4 hospitals, 1222 scans) | SARS-CoV-2 CT-scan (2481 images) |
| Sources | 4 sources, logit-adjusted CE (multi-task) | 1 source → source head **removed**, BCE only |
| Sample unit | 8 KDS-selected slices per scan | **1 image per sample** (KDS removed) |
| Runtime | Google Colab + Google Drive | **Local PyCharm notebook** (auto path detection) |
| Preprocessing | Fixed folder structure + CSVs | Auto-detect class folders, stratified split (seed=42) |
| Training | 8 epochs, fixed LR, save by F1 | 16 epochs, **ReduceLROnPlateau + early stopping**, save by **AUC**, **threshold tuning** |
| Interpretability | Figures only | **Grad-CAM** cell included |
| GPU | NVIDIA A100 (40GB) | Consumer GPU (4GB VRAM, tested: RTX 3050 Ti) |

---

## Results

Trained on RTX 3050 Ti (4GB), 16 configured epochs → **early-stopped at epoch 12** (best checkpoint: epoch 9).

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.60%** (495/497) |
| **F1 Score** | **0.9960** (tuned threshold = 0.26) |
| **AUC-ROC** | **0.9999** |
| **PR-AUC** | 0.9999 |
| **Sensitivity** | **100.00%** (251/251 COVID detected) |
| **Specificity** | 99.19% (244/246) |

```
Confusion Matrix (val = 497 images, threshold 0.26)
              Pred    Non-COVID    COVID
Actual
Non-COVID           244 (TN)       2 (FP)
COVID                 0 (FN)     251 (TP)
```

**Comparison**

| Configuration | F1 | AUC |
|---------------|----|----|
| xDNN baseline (dataset authors) | 0.9731 | – |
| Purdue original (PHAROS, γ=0.5) | 0.9098 | 0.9647 |
| Previous fork run (RICORD) | 0.7879 | 0.8056 |
| **This work (SARS-CoV-2 CT-scan)** | **0.9960** | **0.9999** |

> See [REPORT.md](REPORT.md) for detailed analysis of all metrics, training curves, error analysis and limitations.

---

## Getting Started

### Requirements

| Requirement | Minimum | Tested |
|-------------|---------|--------|
| Python | 3.10+ | 3.12.3 |
| PyTorch | 2.1+ | 2.5.1+cu121 |
| GPU VRAM | 4GB (batch 8) | RTX 3050 Ti 4GB |
| Disk space | ~1GB (dataset 242MB + preprocessed) | – |
| IDE | PyCharm Professional (Jupyter built-in) | – |

> ⚠️ **PyCharm Community Edition cannot open `.ipynb` notebooks.** In that case, run the notebook with `jupyter notebook` in a browser, or ask for a `.py` conversion with `# %%` cell markers.

### Step 1: Clone and Setup Environment

```bash
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct

# PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Other dependencies (the notebook also auto-installs anything missing)
pip install timm albumentations scikit-learn opencv-python pandas matplotlib scipy tqdm
```

> ⚠️ Install `torchvision` **pinned to your torch version** — a newer `torchvision` from PyPI will silently replace your CUDA torch with a CPU build. The notebook's first cell contains an `ensure_torchvision()` guard against this.

**Verify:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu121 True
```

### Step 2: Download the Dataset

**Option A — Kaggle CLI (recommended):**
```bash
pip install kaggle
# Create API token: https://www.kaggle.com/settings -> API -> Create New Token
# Place kaggle.json in C:\Users\<user>\.kaggle\ (Windows) or ~/.kaggle/ (Linux)
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p data --unzip
```

**Option B — Manual download:**

Download from [Kaggle dataset page](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) and extract into `data/` so that two image folders exist (any nesting depth is auto-detected):

```
data/
├── COVID/            # 1252 CT images  (or CT_COVID/)
└── non-COVID/        # 1230 CT images  (or CT_NonCOVID/)
```

**Time**: ~5 minutes (242 MB)

### Step 3: Run the Notebook in PyCharm

1. **File → Open** → select this repository root (`multisource-covid-ct/`)
2. Select the Python interpreter used in Step 1
3. Open **`MultiSource_COVID_CT.ipynb`** → **Run All** (top to bottom)

The notebook:

| Cell | What it does |
|------|--------------|
| 1 | Auto-install missing packages, paths, hyperparameters, device |
| 2 | Locate the two class folders (auto-detect, excludes preprocessed data) |
| 3 | SSFL lung-extraction functions |
| 4 | Stratified 80/20 split (seed 42) + preprocess to `data/preprocessed_sarscov2/` (256×256) |
| 5 | Augmentations, Dataset, EfficientNet-B7 (binary head), train/val loops |
| 6 | Build train/val lists |
| 7 | **Training**: Adam + ReduceLROnPlateau + early stopping, checkpoint by AUC |
| 8 | **Threshold tuning** (sweep 0.05–0.95 on val) + quick metrics |
| 9 | Full report: confusion matrix, ROC/PR, curves, per-class F1 → `results/evaluation_plots.png` |
| 10 | Save `val_predictions.csv`, weights, `results.txt` |
| 11 | **Grad-CAM** visualization → `results/gradcam.png` |

**Training time**: ~2–3 min/epoch on RTX 3050 Ti (batch 8) → ~30–40 minutes total (less with early stopping).

### Outputs

```
checkpoints/effnet_best.pth       # best model (by val AUC)
results/
├── evaluation_plots.png          # confusion matrix, ROC, PR, curves
├── gradcam.png                   # Grad-CAM overlays (6 correct + 3 wrong)
├── val_predictions.csv           # per-image predictions
└── results.txt                   # metrics + epoch history
```

---

## Quick Start

```bash
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scikit-learn opencv-python pandas matplotlib scipy tqdm

kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p data --unzip

# Then open MultiSource_COVID_CT.ipynb in PyCharm and Run All
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Lower `BATCH` to 4 (or 2) in cell 1; or set `MODEL_NAME='tf_efficientnet_b3'` |
| `torch.cuda.is_available() == False` after pip install | A package upgraded torch to CPU build — restore with `pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121` |
| `AttributeError: ... no attribute 'act1'` | Make sure the notebook uses `conv_head` (timm ≥ 1.0 removed `act1`) |
| `Khong tim thay CT_COVID/ va CT_NonCOVID/` | Dataset not under `data/` — check Step 2 folder names |
| Notebook won't open in PyCharm | Community Edition doesn't support `.ipynb` — use PyCharm Professional or Jupyter |
| Kernel not found in PyCharm | Settings → Jupyter → select the interpreter from Step 1 |
| `CoarseDropout ... unexpected keyword` | albumentations < 1.4 API — upgrade: `pip install -U albumentations` |
| Training too slow / CPU only | Check `device:` printed in cell 1 must be `cuda` |
| Suspiciously perfect metrics | See Limitations in REPORT.md (image-level split, no patient IDs) |

---

## Project Structure

```
├── MultiSource_COVID_CT.ipynb    # ★ End-to-end notebook (PyCharm / Jupyter)
├── REPORT.md                     # Detailed project report (Vietnamese)
├── data/
│   ├── README.md                 # Dataset download & structure
│   ├── COVID/                    # Raw dataset (not tracked)
│   ├── non-COVID/                # Raw dataset (not tracked)
│   └── preprocessed_sarscov2/    # After preprocessing (not tracked)
├── checkpoints/                  # Model weights (not tracked)
├── results/                      # Metrics, plots, Grad-CAM (not tracked)
├── configurations/
│   └── default.yaml              # Hyperparameters (original pipeline)
├── scripts/                      # Original + RICORD pipeline (legacy)
│   ├── convert_dicom_to_png.py   # DICOM → PNG (RICORD experiment)
│   ├── create_csv_splits.py      # CSV splits (RICORD experiment)
│   ├── sweep_gamma.sh            # γ sweep (original)
│   └── visualize_results.py      # Paper figures (original)
├── src/                          # Original multi-task library (legacy)
│   ├── model.py                  # Multi-task EfficientNet-B7
│   ├── losses.py                 # BCE + Logit-Adjusted CE
│   ├── dataset.py
│   ├── preprocessing.py
│   └── engine.py
├── train.py                      # Original CLI training (PHAROS/RICORD)
├── preprocess.py                 # Original CLI preprocessing
├── evaluate.py                   # Original CLI evaluation
├── inference.py
├── requirements.txt
├── LICENSE
└── README.md
```

> The notebook is self-contained: it does **not** depend on `src/`, `train.py` or the RICORD scripts — those remain from previous experiments (single-source RICORD run, see git history).

---

## Method

1. **Preprocessing** — SSFL lung extraction (spatial filtering, binarization, morphological closing) isolates the lung region; every image resized to 256×256. KDS slice sampling from the original pipeline is dropped because each sample here is a single 2D image.

2. **Architecture** — ImageNet-pretrained EfficientNet-B7 produces a 2560-dim feature vector per image; a dropout(0.3) + linear head outputs a single logit. The 4-class source head is removed (dataset has no hospital labels).

3. **Training** — BCE loss, Adam (lr=1e-4, wd=5e-4), AMP fp16, batch 8. `ReduceLROnPlateau` (factor 0.5, patience 1, on val AUC), early stopping (patience 3), checkpoint selected by **val AUC** (max 16 epochs).

4. **Decision threshold** — swept over 0.05–0.95 on validation to maximize F1 (result: 0.26 instead of the default 0.50).

5. **Interpretability** — Grad-CAM on `conv_head` highlights the lung regions driving each prediction.

---

## Report

See [REPORT.md](REPORT.md) (Vietnamese) for detailed analysis including:
- Dataset statistics and verification (duplicates, split balance)
- Every metric explained with clinical interpretation
- Epoch-by-epoch training analysis (early stopping, scheduler)
- Error analysis (the 2 misclassified images)
- Comparison with baselines and previous experiments
- Limitations (image-level split, no patient IDs) and future work

---

## Acknowledgements

This work builds upon the multi-source COVID-19 detection framework developed by the M2 Lab at Purdue University, supported by the U.S. National Science Foundation (NSF) under grant IIS-2434967, and the National Artificial Intelligence Research Resource (NAIRR) Pilot and TACC Lonestar6.

Dataset: [SARS-CoV-2 CT-scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) (Soares et al., 2020, CC BY-NC-SA 4.0).

## License

This project is released under the [MIT License](LICENSE).
