# Báo Cáo Dự Án: Phân Loại COVID-19 CT với EfficientNet-B7

## Mục Lục

- [1. Tổng Quan](#1-tổng-quan)
- [2. Tập Dữ Liệu](#2-tập-dữ-liệu)
- [3. Quy Trình Xử Lý](#3-quy-trình-xử-lý)
- [4. Kết Quả](#4-kết-quả)
- [5. Giải Thích Các Chỉ Số Đánh Giá](#5-giải-thích-các-chỉ-số-đánh-giá)
- [6. Phân Tích Quá Trình Training](#6-phân-tích-quá-trình-training)
- [7. Phân Tích Lỗi](#7-phân-tích-lỗi)
- [8. So Sánh](#8-so-sánh)
- [9. Hạn Chế Và Hướng Cải Thiện](#9-hạn-chế-và-hướng-cải-thiện)
- [10. Tài Liệu Tham Khảo](#10-tài-liệu-tham-khảo)

---

## 1. Tổng Quan

Dự án thích ứng khung phát hiện COVID-19 CT từ [Purdue-M2/multisource-covid-ct](https://github.com/Purdue-M2/multisource-covid-ct) để huấn luyện trên **tập SARS-CoV-2 CT-scan**, chạy hoàn toàn **local trên PyCharm** bằng notebook `MultiSource_COVID_CT.ipynb`.

### Thông Tin Chính

| Mục | Chi Tiết |
|-----|----------|
| **Bài báo gốc** | "Robust Multi-Source COVID-19 Detection in CT Images" — AIMS @ CVPR 2026 |
| **Tác giả (gốc)** | Pritha, Xu, Ding, Li, Hou, Wang, Hu — M2 Lab, Purdue University |
| **Backbone** | EfficientNet-B7 (pretrained ImageNet, đặc trưng 2560-dim → 1 logit) |
| **Nhiệm vụ** | Phân loại nhị phân: COVID-19 dương tính vs âm tính |
| **Tập dữ liệu** | SARS-CoV-2 CT-scan (Kaggle, Soares et al. 2020) — 2481 ảnh |
| **Môi trường** | Python 3.12.3, PyTorch 2.5.1+cu121, RTX 3050 Ti (4GB) |
| **Chế độ** | Đơn nguồn — **bỏ source head**, chỉ BCE loss |

### Thay Đổi So Với Phiên Bản Gốc

| Khía Cạnh | Gốc (PHAROS) | Dự Án Này (SARS-CoV-2 CT-scan) |
|-----------|--------------|--------------------------------|
| Tập dữ liệu | PHAROS đa nguồn (4 bệnh viện, 1222 scans) | SARS-CoV-2 CT-scan (2481 ảnh, Brazil) |
| Nhãn nguồn | 4 nguồn + logit-adjusted CE (multi-task) | Không có → **bỏ source head**, chỉ BCE |
| Đơn vị mẫu | 8 slice KDS mỗi scan | **1 ảnh / mẫu** (bỏ KDS) |
| Chạy | Colab + Google Drive | **PyCharm notebook**, tự dò đường dẫn |
| Chia tập | Theo scan (CSV) | Stratified 80/20 theo ảnh, seed=42 |
| Huấn luyện | 8 epoch, LR cố định, lưu theo F1 | **16 epoch, ReduceLROnPlateau + early stopping, lưu theo AUC, tune ngưỡng** |
| Giải thích | Chỉ hình ảnh pipeline | Thêm **Grad-CAM** |
| GPU | NVIDIA A100 (40GB) | RTX 3050 Ti (4GB) |

---

## 2. Tập Dữ Liệu

### 2.1 Tổng Quan

**[SARS-CoV-2 CT-scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)** — CT ngực thật từ các bệnh viện tại São Paulo, Brazil (Soares, Angelov et al., 2020).

| Thông Tin | Giá Trị |
|-----------|---------|
| Số ảnh chính thức | 2482 (1252 COVID + 1230 Non-COVID) |
| Số ảnh trên máy (đã kiểm tra) | **2481** (1252 + 1229 — thiếu 1 ảnh, không đáng kể) |
| Dung lượng | ~242 MB, định dạng PNG |
| Kích thước ảnh | Không đồng đều 146–490 px (đã resize 256×256) |
| Giấy phép | CC BY-NC-SA 4.0 |
| Baseline chính thức | xDNN — F1 = 97.31% |

### 2.2 Kiểm Tra Chất Lượng Dữ Liệu

| Kiểm Tra | Kết Quả |
|----------|---------|
| Đọc ảnh lỗi (quét 80 ảnh ngẫu nhiên) | **0 lỗi** |
| Ảnh trùng lặp hoàn toàn (cùng MD5) | **2 ảnh** (0.08%) |
| Ảnh trùng lặp qua 2 lớp (nghiêm trọng) | **0** ✓ |
| Cặp trùng lặp nằm xen train/val | **1 cặp** → rò rỉ ~0.2% tập val (hạn chế, xem Mục 9) |

### 2.3 Chia Train/Val (Stratified 80/20, Seed 42)

| Tập | COVID | Non-COVID | Tổng |
|-----|-------|-----------|------|
| Train | 1001 | 983 | **1984** |
| Validation | 251 | 246 | **497** |
| **Tổng** | 1252 | 1229 | **2481** |

Tỷ lệ lớp gần cân bằng tuyệt đối (50.4% / 49.6%) — không cần xử lý imbalance.

---

## 3. Quy Trình Xử Lý

### 3.1 Kiến Trúc

```
Input: 1 ảnh CT (3×256×256, ImageNet-normalized)
    ↓
EfficientNet-B7 (ImageNet pretrained)
    ↓ feature vector (dim=2560)
    ↓
Dropout(0.3) → Linear(2560 → 1)
    ↓ logit
    ↓ sigmoid → xác suất COVID
```

- **Bỏ hoàn toàn source head** (4 lớp bệnh viện) — tập dữ liệu không có nhãn nguồn
- Ảnh xám được chuyển RGB, mỗi ảnh suy luận riêng (không pooling 8 slice)

### 3.2 Tiền Xử Lý

1. **Chia tập trước** — stratified 80/20 (seed=42) để tránh rò rỉ qua tiền xử lý
2. **SSFL Lung Extraction**: lọc không gian (uniform + minimum filter) → ngưỡng Otsu → contour → đóng hình thái → cắt viền → resize 256×256
3. **Bỏ KDS Sampling** — không còn khái niệm "8 slice/scan" vì mỗi mẫu là 1 ảnh
4. Toggle `USE_LUNG_EXTRACT = False` để chạy ablation trên ảnh gốc

### 3.3 Cấu Hình Training

| Tham Số | Giá Trị |
|---------|---------|
| Backbone | EfficientNet-B7 (pretrained, ~66M tham số) |
| Optimizer | Adam (lr=1e-4, weight_decay=5e-4) |
| Batch size | 8 (đo VRAM: 3.01/4.00 GiB) |
| Epochs | 16 (cấu hình) — **dừng sớm ở epoch 12** |
| Mixed precision | AMP fp16 |
| Loss | BCEWithLogitsLoss |
| LR scheduler | **ReduceLROnPlateau** (mode=max, factor=0.5, patience=1, min_lr=1e-6) theo val AUC |
| Early stopping | **Patience 3** trên val AUC |
| Checkpoint | Lưu theo **val AUC** (ổn định hơn F1) |
| Ngưỡng quyết định | **Tune 0.05–0.95 trên val → 0.26** (mặc định 0.50) |
| num_workers | 0 trên Windows (tránh treo kernel) |

### 3.4 Tăng Cường Dữ Liệu (albumentations 2.0.8)

| Biến Đổi | Tham Số |
|----------|--------|
| Resize | 256 × 256 |
| HorizontalFlip | p=0.5 |
| Affine (thay ShiftScaleRotate) | translate ±20%, scale 0.8–1.2, rotate ±30°, p=0.5 |
| HueSaturationValue | shift 20/20/20, p=0.5 |
| RandomBrightnessContrast | 0.2/0.2, p=0.5 |
| CoarseDropout | 1–8 lỗ, kích thước 8–32 px, p=0.2 |
| Normalize | ImageNet mean/std |

### 3.5 Quy Trình Hoàn Chỉnh

```bash
# Bước 1: Clone + cài đặt
git clone https://github.com/BaoNgo355/multisource-covid-ct.git
cd multisource-covid-ct
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scikit-learn opencv-python pandas matplotlib scipy tqdm

# Bước 2: Tải dataset (Kaggle CLI hoặc tải tay → để vào data/)
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p data --unzip

# Bước 3: Mở MultiSource_COVID_CT.ipynb trong PyCharm → Run All
```

---

## 4. Kết Quả

### 4.1 Đánh Giá Cuối Cùng (Best Checkpoint: Epoch 9, Early Stop tại Epoch 12)

| Chỉ Số | Giá Trị |
|--------|---------|
| **Accuracy** | **99.60%** (495/497) |
| **F1 Score** | **0.9960** (ngưỡng tối ưu 0.26) |
| **AUC-ROC** | **0.9999** |
| **PR-AUC** | 0.9999 |
| **Sensitivity** | **100.00%** |
| **Specificity** | 99.19% |

### 4.2 Ma Trận Nhầm Lẫn (Ngưỡng 0.26)

```
                Dự Đoán
                Non-COVID    COVID
Thực Tế
Non-COVID          244 (TN)    2 (FP)
COVID                0 (FN)  251 (TP)
```

| Thành Phần | Số Lượng | Mô Tả |
|------------|----------|-------|
| **TN = 244** | 244 | Non-COVID nhận diện đúng |
| **TP = 251** | 251 | COVID nhận diện đúng — **không bỏ sót ca nào** |
| **FP = 2** | 2 | Non-COVID báo nhầm COVID |
| **FN = 0** | 0 | Không có false negative |

### 4.3 Hiệu Suất Theo Lớp

| Lớp | Precision | Recall | F1-Score | Số Mẫu |
|-----|-----------|--------|----------|--------|
| Non-COVID | 1.0000 | 0.9919 | 0.9959 | 246 |
| COVID | 0.9921 | 1.0000 | 0.9960 | 251 |
| **Trung bình** | **0.9960** | **0.9960** | **0.9960** | **497** |

### 4.4 Phân Bố Xác Suất (Rất Tách Biệt)

| Lớp | Trung vị | 25%–75% | Min–Max |
|-----|----------|---------|---------|
| COVID (đúng nhãn) | **1.000** | 0.999–1.000 | 0.451–1.000 |
| Non-COVID | **0.001** | 0.000–0.008 | 0.000–0.772 |

### 4.5 Tune Ngưỡng Quyết Định

Quét ngưỡng 0.05–0.95 trên tập validation để tối ưu F1:

| Ngưỡng | F1 (validation, best checkpoint) |
|--------|----------------------------------|
| 0.50 (mặc định) | 0.9940 |
| **0.26 (tối ưu)** | **0.9960** |

→ Mặc dù margin đã lớn, chọn ngưỡng 0.26 vẫn cải thiện F1 và giữ nguyên Sensitivity = 100%.
Ngưỡng được lưu vào `results/results.txt` để inference dùng thống nhất (không dùng 0.5 lung tung giữa các bảng).

---

## 5. Giải Thích Các Chỉ Số Đánh Giá

### 5.1 F1 Score = 0.9960

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Precision (COVID)** = 251/253 = 0.992 → Khi model nói "COVID", đúng 99.2%
- **Recall (COVID)** = 251/251 = 1.000 → Phát hiện **100%** ca COVID
- **F1** = trung bình điều hòa ≈ 0.996

**Giải thích**: Cả precision lẫn recall đều ~1 → model cân bằng tuyệt vời giữa bỏ sót và báo giả.

### 5.2 AUC-ROC = 0.9999

- Nếu chọn ngẫu nhiên 1 ảnh COVID và 1 ảnh Non-COVID, model **xếp đúng thứ tự 99.99%** số lần
- Gần như hoàn hảo — nhưng lưu ý đây là tập val **theo ảnh** (xem Mục 9)

### 5.3 Sensitivity = 100%

- 251 ca COVID → model phát hiện **cả 251**, bỏ sót **0**
- **Đây là chỉ số quan trọng nhất cho sàng lọc** — mẫu vật tế nhị nhất trong báo cáo

### 5.4 Specificity = 99.19%

- 246 Non-COVID → đúng 244, báo giả 2
- Ít cảnh báo giả → giảm gánh nặng kiểm tra lại

### 5.5 Accuracy = 99.60%

```
Accuracy = (251 + 244) / 497 = 495 / 497 = 99.60%
```

### 5.6 Bảng Tổng Hợp

| Chỉ Số | Giá Trị | Yêu Cầu Lâm Sàng | Trạng Thái |
|--------|---------|-------------------|------------|
| Sensitivity | 100.00% | >90% | ✅ Đạt |
| Specificity | 99.19% | >85% | ✅ Đạt |
| F1 Score | 0.9960 | >0.85 | ✅ Đạt |
| AUC-ROC | 0.9999 | >0.85 | ✅ Đạt |

---

## 6. Phân Tích Quá Trình Training

### 6.1 Lịch Sử Training (16 cấu hình → dừng ở epoch 12)

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Val AUC | Ghi Chú |
|-------|-----------|----------|----------|--------|---------|---------|
| 1 | 0.4351 | 0.7924 | 0.2589 | 0.9157 | 0.9737 | Ban đầu |
| 2 | 0.2868 | 0.8829 | 0.1542 | 0.9679 | 0.9941 | Tăng nhanh |
| 3 | 0.2455 | 0.9075 | 0.0900 | 0.9658 | 0.9957 | |
| 4 | 0.1839 | 0.9306 | 0.1244 | 0.9695 | 0.9949 | |
| 5 | 0.1588 | 0.9459 | 0.0682 | 0.9741 | 0.9980 | |
| 6 | 0.1699 | 0.9418 | 0.0707 | 0.9820 | 0.9964 | |
| 7 | 0.1185 | 0.9553 | 0.0725 | 0.9763 | 0.9966 | |
| 8 | 0.0718 | 0.9744 | 0.0365 | 0.9820 | 0.9993 | |
| **9** | 0.0450 | 0.9825 | **0.0208** | **0.9940** | **0.9999** | **★ Best — checkpoint lưu tại đây** |
| 10 | 0.0757 | 0.9730 | 0.0308 | 0.9921 | 0.9996 | Không vượt epoch 9 |
| 11 | 0.0890 | 0.9725 | 0.0401 | 0.9842 | 0.9993 | Không vượt (2 liên tiếp) |
| 12 | 0.0431 | 0.9860 | 0.0448 | 0.9862 | 0.9995 | Không vượt (3 liên tiếp) → **Early Stop** |

### 6.2 Quan Sát

1. **Epoch 1–9**: hội tụ nhanh và ổn định — val AUC đi từ 0.9737 → 0.9999
2. **Val loss đáy ở epoch 9** (0.0208) rồi tăng nhẹ (0.0308 → 0.0448) → **overfit bắt đầu**
3. **Early stopping hoạt động đúng**: đủ 3 epoch không cải thiện AUC → dừng tại epoch 12/16, tiết kiệm ~10 phút
4. **Checkpoint theo AUC** chọn đúng epoch 9 — không phải epoch có F1 cao nhất (12) → đánh giá ổn định hơn
5. **Train F1 < Val F1** là bình thường: train đo trong lúc augment + dropout, val đo sạch
6. **Scheduler ReduceLROnPlateau**: LR tự giảm khi AUC chững → 2 epoch cuối vẫn giữ AUC ~0.9995

---

## 7. Phân Tích Lỗi

### 7.1 False Negatives (FN = 0)

**Không có** — model không bỏ sót ca COVID nào trong 497 ảnh val.

### 7.2 False Positives (FP = 2)

| Ảnh | Xác Suất COVID | Nhận Định |
|-----|----------------|-----------|
| `Non-Covid (254).png` | 0.707 | Gần ranh giới nhưng vẫn cao — có thể có tổn thương nhìn giống COVID |
| `Non-Covid (685).png` | 0.772 | Tương tự — nên xem lại ảnh gốc và nguồn dán nhãn |

**Nguyên nhân có thể:**
- Bệnh lý phổi khác (viêm phổi khác nguyên nhân, phù phổi) giống mô hình COVID
- Ảnh gần như "xấp xỉ COVID" — y tế thực tế cần bác sĩ đọc lại (đúng quy trình)
- Một phần do **rò rỉ nhẹ**: 1 cặp ảnh trùng lặp nằm xen train/val

### 7.3 Grad-CAM (Đính Kèm `results/gradcam.png`)

- Layer hook: `conv_head` (conv cuối EfficientNet, timm 1.0.x)
- Hiển thị 6 ảnh đúng + 3 ảnh sai
- Quan sát: heatmap bám **vùng phổi** (không phải viền/chữ) → model học được đặc trưng y tế thật
- Ảnh dự đoán sai thường có vùng nóng rải khắp 2 phổi thay vì tổn thương khu trú

---

## 8. So Sánh

### 8.1 Với Các Baseline

| Chỉ Số | xDNN (baseline gốc) | Purdue (PHAROS, γ=0.5) | Fork trước (RICORD) | **Dự Án Này** |
|--------|---------------------|------------------------|---------------------|---------------|
| Tập dữ liệu | SARS-CoV-2 CT-scan | PHAROS (4 nguồn) | RICORD (183 scans) | **SARS-CoV-2 CT-scan** |
| F1 | 0.9731 | 0.9098 | 0.7879 | **0.9960** |
| AUC-ROC | – | 0.9647 | 0.8056 | **0.9999** |
| Accuracy | – | 92.53% | 79.40% | **99.60%** |
| GPU | – | A100 (40GB) | 4GB GPU | 4GB GPU |

**Nhận xét:**
- Vượt **baseline xDNN của chính tác giả dataset** (+2.3% F1)
- Cao hơn cả kết quả gốc trên PHAROS — nhờ dataset này "dễ" hơn (phân tách rõ) và transfer learning tốt
- ⚠️ Nhưng so sánh **không công bằng tuyệt đối**: các con số dùng giao thức chia tập khác nhau (xem Mục 9)

### 8.2 So Với Phiên Bản Gốc Của Repo

| Khía Cạnh | Purdue (PHAROS) | Dự Án Này |
|-----------|-----------------|-----------|
| F1 / AUC | 0.9098 / 0.9647 | 0.9960 / 0.9999 |
| Mục tiêu | Đa nhiệm (phân loại + nhận diện nguồn) | Nhị phân thuần |
| Sample | Scan 8-slice | 1 ảnh |
| Huấn luyện | 8 epoch, không scheduler | 16 epoch, scheduler + early stop |

---

## 9. Hạn Chế Và Hướng Cải Thiện

### 9.1 Hạn Chế Hiện Tại

| Hạn Chế | Ảnh Hướng | Ưu Tiên |
|---------|-----------|---------|
| **Chia tập theo ảnh, không theo bệnh nhân** (dataset không có mã BN) | Có thể cùng bệnh nhân ở cả 2 tập → **số liệu val lạc quan** | **Cao** |
| 1 cặp ảnh trùng lặp xen train/val | Rò rỉ ~0.2% tập val | Trung bình |
| Tune ngưỡng trên chính tập val | F1 tối ưu hơi lạc quan (một phần do val làm cả việc chọn model + chọn ngưỡng) | Trung bình |
| Một lần split duy nhất (seed 42) | Số liệu phụ thuộc may mắn của lần chia | Trung bình |
| Không có nhãn nguồn bệnh viện | Mất phần multi-task (logit adjustment) — mục tiêu ban đầu của bài gốc | Theo đề bài |
| Dataset đơn vùng (Brazil, 2020) | Generalization sang nơi khác chưa kiểm chứng | Thấp |

### 9.2 Hướng Cải Thiện

| Cải Thiện | Hiệu Quả Dự Kiến | Độ Khó |
|-----------|-------------------|--------|
| **5-fold cross-validation** | Số liệu trung bình ± độ lệch chuẩn, vững hơn cho phản biện | Trung bình (3–4 tiếng chạy) |
| **Deduplicate trước khi split** | Loại bỏ hoàn toàn rò rỉ trùng lặp | Dễ |
| **Ablation: B3 vs B7, có/không lung extract, có/không augment** | Bảng thành phần cho báo cáo | Dễ (chỉ chạy lại) |
| **Giữ nguyên tập test riêng** (tách val làm 2: tune/eval) | Ước tính trung thực hơn | Trung bình |
| TTA (test-time augmentation) | +0.5–1% F1 | Dễ |
| Gộp thêm dataset 4173 ảnh (cùng nhóm tác giả) | Thêm dữ liệu train | Dễ |
| Grad-CAM nâng cao (per-layer so sánh) | Phần thảo luận phong phú | Trung bình |

---

## 10. Tài Liệu Tham Khảo

1. **Bài báo gốc**: Pritha et al., "Robust Multi-Source COVID-19 Detection in CT Images," CVPR 2026 Workshop on New Trends in AI-Generated Media and Security (AIMS).
2. **Code gốc**: https://github.com/Purdue-M2/multisource-covid-ct
3. **Tập dữ liệu**: Soares, Angelov, Biaso, Higa Froes, Kanda Abe, "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification," medRxiv 2020. doi:10.1101/2020.04.24.20078584
4. **Baseline xDNN**: Angelov & Soares, "Towards explainable neural networks (xDNN)," Neural Networks 130 (2020) 185–194.
5. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019.
6. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.

---

*Báo cáo được tạo: Tháng 10, 2026*
*Mô hình: `checkpoints/effnet_best.pth` (EfficientNet-B7, epoch 9, val AUC 0.9999)*
*Repository: https://github.com/BaoNgo355/multisource-covid-ct*
