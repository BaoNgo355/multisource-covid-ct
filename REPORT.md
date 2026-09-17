# Báo cáo Dự án: Phân loại COVID-19 trên ảnh CT

## Robust Multi-Source COVID-19 Detection in CT Images
### Phiên bản: RICORD Single-Source Training

---

## 1. Tổng quan dự án

### 1.1 Mục tiêu
Xây dựng mô hình phân loại ảnh CT ngực để phát hiện COVID-19, sử dụng EfficientNet-B7 backbone với multi-task learning framework.

### 1.2 Nguồn gốc
- **Repository gốc**: [Purdue-M2/multisource-covid-ct](https://github.com/Purdue-M2/multisource-covid-ct)
- **Paper**: "Robust Multi-Source COVID-19 Detection in CT Images" - CVPR 2026 AIMS Workshop
- **Tác giả**: Asmita Yuki Pritha, Jason Xu, Daniel Ding, Justin Li, Aryana Hou, Xin Wang, Shu Hu (M2 Lab, Purdue University)
- **Phiên bản adaptor**: BaoNgo355/multisource-covid-ct

### 1.3 Thay đổi so với gốc

| Aspect | Gốc (PHAROS) | Phiên bản này (RICORD) |
|--------|-------------|------------------------|
| Dataset | PHAROS Multi-Source (4 bệnh viện) | RICORD-1A + 1B (1 nguồn) |
| Số nguồn | 4 sources | 1 source |
| Loss function | Logit-Adjusted CE (γ=0.5) | BCE only (γ=0.0) |
| Định dạng dữ liệu | PNG sẵn có | DICOM → PNG |
| Preprocessing | Folder structure cố định | CSV-based splits |
| GPU | NVIDIA A100 (40GB) | GPU consumer (4GB VRAM) |
| Epochs | 8 | 20 |
| Batch size | 10 | 2 |

---

## 2. Dataset

### 2.1 Nguồn dữ liệu
- **RICORD-1A** (COVID-Positive): Download từ TCIA (The Cancer Imaging Archive)
- **RICORD-1B** (COVID-Negative): Download từ TCIA
- Công cụ tải: TCIA Data Retriever

### 2.2 Thống kê dữ liệu

| Collection | DICOM Series | Scans hợp lệ | Tổng slices |
|------------|-------------|--------------|-------------|
| RICORD-1A (COVID+) | 110 | 89 | 11,797 |
| RICORD-1B (COVID−) | 117 | 79 | 12,259 |
| **Tổng** | **227** | **168** | **24,056** |

### 2.3 Chia dữ liệu

| Split | COVID | Non-COVID | Tổng |
|-------|-------|-----------|------|
| Train (80%) | 71 | 63 | 134 |
| Validation (20%) | 18 | 16 | 34 |

---

## 3. Pipeline xử lý

### 3.1 DICOM → PNG Conversion
- **Lung Windowing**: W=1500, L=-600
- **Loại trừ series**: SCOUT, COR 3X3, SAG 3X3, bone algorithms
- **Ưu tiên series**: ROUTINE CHEST NON-CON
- **Script**: `scripts/convert_dicom_to_png.py`

### 3.2 Train/Val Splits
- **Tỷ lệ**: 80/20 stratified
- **Định dạng**: CSV files (ct_scan_name, data_centre)
- **Script**: `scripts/create_csv_splits.py`

### 3.3 Preprocessing
- **SSFL Lung Extraction**: Phân đoạn phổi qua spatial filtering, binarization, morphological closing
- **KDS Sampling**: Gaussian KDE để chọn 8 slices đại diện per scan
- **Kích thước ảnh**: 256×256

### 3.4 Diagram Pipeline

```
DICOM Files (TCIA)
    ↓ convert_dicom_to_png.py
PNG Images (lung windowing)
    ↓ create_csv_splits.py
Train/Val CSV Splits
    ↓ preprocess.py (SSFL + KDS)
Preprocessed Scans (8 slices/scan, 256×256)
    ↓ train.py
EfficientNet-B7 Model
    ↓ evaluate.py
Kết quả đánh giá
```

---

## 4. Kiến trúc mô hình

### 4.1 Backbone
- **EfficientNet-B7** (pretrained on ImageNet)
- 66M parameters, output feature dimension: 2560

### 4.2 Multi-Task Architecture
```
Input: 8 CT slices (256×256)
    ↓ EfficientNet-B7 (per slice)
8 feature vectors (dim=2560)
    ↓ Element-wise Mean Pooling
Scan-level representation (dim=2560)
    ├── COVID-19 Classifier (Binary) → Logits
    └── Source Identifier (1-class) → [Not used in γ=0.0]
```

### 4.3 Loss Function
- **COVID Head**: BCEWithLogitsLoss
- **Source Head**: LogitAdjustedCE (disabled when γ=0.0)
- **Total Loss**: ℓ = ℓ_BCE + γ · ℓ_LA

### 4.4 Training Configuration

```yaml
# configurations/default.yaml
backbone: tf_efficientnet_b7
num_sources: 1
dropout: 0.3
image_size: 256
num_slices: 8
gamma: 0.0
epochs: 20
batch_size: 2
learning_rate: 1.0e-4
weight_decay: 5.0e-4
precision: amp
```

---

## 5. Kết quả

### 5.1 Final Evaluation Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.7879** |
| **AUC-ROC** | **0.8056** |
| **Competition Score** | **0.7939** |
| **Accuracy** | 79.4% |
| **Sensitivity (Recall COVID)** | 72.2% |
| **Specificity (Recall Non-COVID)** | 87.5% |

### 5.2 Confusion Matrix

```
              Predicted
              Non-COVID    COVID
Actual
Non-COVID        14 (TN)     2 (FP)
COVID             5 (FN)    13 (TP)
```

### 5.3 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-COVID | 0.74 | 0.88 | 0.80 | 16 |
| COVID | 0.87 | 0.72 | 0.79 | 18 |
| **Weighted Avg** | **0.81** | **0.79** | **0.79** | **34** |

### 5.4 Training History

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Val AUC |
|-------|-----------|----------|----------|--------|---------|
| 1 | 0.6906 | 0.5401 | 0.7145 | 0.3704 | 0.5208 |
| 2 | 0.6876 | 0.6164 | 0.6347 | 0.5926 | 0.8264 |
| 14 (best) | 0.5762 | 0.7273 | 0.6338 | 0.7333 | 0.7674 |
| 15 | 0.5004 | 0.7660 | 0.6892 | 0.7059 | 0.7500 |
| 16 | 0.4113 | 0.8116 | 0.9960 | 0.5714 | 0.6840 |
| 17 | 0.3709 | 0.8531 | 0.9003 | 0.6471 | 0.7361 |
| 18 | 0.3008 | 0.8759 | 1.0775 | 0.7273 | 0.7812 |
| 19 | 0.2602 | 0.9078 | 0.9189 | 0.6897 | 0.7778 |
| 20 | 0.4150 | 0.8571 | 0.7570 | 0.6154 | 0.7778 |

**Best Model**: Checkpoint tại epoch 14 (selected by validation AUC)

---

## 6. Phân tích

### 6.1 Điểm mạnh
- **Specificity cao (87.5%)**: Model nhận diện Non-COVID tốt, ít false positive
- **Precision COVID cao (87%)**: Khi model dự đoán COVID, độ tin cậy cao
- **AUC-ROC tốt (0.8056)**: Model có khả năng phân biệt giữa 2 lớp ở các ngưỡng khác nhau

### 6.2 Điểm yếu
- **Sensitivity thấp hơn (72.2%)**: 5/18 COVID cases bị bỏ sót (FN)
- **Overfitting rõ rệt**: Train loss giảm mạnh (0.69→0.26), Val loss dao động tăng
- **Val metrics không ổn định**: Val AUC dao động 0.68-0.78 qua các epochs

### 6.3 Overfitting Analysis

```
Gap Train-Val Loss:
Epoch  1: 0.6906 - 0.7145 = -0.024 (gap nhỏ)
Epoch 14: 0.5762 - 0.6338 = -0.058 (gap vừa)
Epoch 18: 0.3008 - 1.0775 = -0.777 (gap lớn → overfitting)
Epoch 20: 0.4150 - 0.7570 = -0.342 (gap lớn)
```

**Nguyên nhân**: Dataset quá nhỏ (134 training scans) cho EfficientNet-B7 (66M params).

---

## 7. So sánh với kết quả gốc (PHAROS)

| Metric | PHAROS (γ=0.5) | RICORD (γ=0.0) | Ghi chú |
|--------|----------------|----------------|---------|
| F1 | 0.9098 | 0.7879 | −12.2% |
| AUC | 0.9647 | 0.8056 | −15.9% |
| Score | 0.8194 | 0.7939 | −2.5% |
| Training scans | 1,222 | 134 | −89% |

**Nhận xét**: Kết quả thấp hơn đáng kể do dataset nhỏ hơn nhiều (134 vs 1,222 scans).

---

*Báo cáo được tạo: September 17, 2026*
