# Kaggle Hedge Fund Time-Series Forecasting

Dự án xây dựng pipeline dự báo chuỗi thời gian đa khung thời gian (Multi-Horizon: $H \in \{1, 3, 10, 25\}$) cho quỹ đầu cơ định lượng, sử dụng LightGBM và cơ chế phòng vệ Dynamic Embargo.

**Tác giả:** Lê Văn Hiệp, Nguyễn Duy Hiếu, Vũ Minh Hiếu, Trần Trung Hiếu

---

## Kết quả

- **Private Leaderboard:** **0.2475** (Weighted RMSE)
- **Out-of-Sample CV:** **0.2626**

---

## Thách thức Dữ liệu

1. **Phân phối Leptokurtic & Ngoại lệ cực trị:** Biến mục tiêu tập trung dày đặc quanh 0 nhưng có các giá trị ngoại lệ lớn từ -2201 đến +2314. Nếu không kiểm soát, cây quyết định sẽ lùi toàn bộ dự báo về 0 để tránh bị phạt (sụp đổ phương sai / zero-inflation).
2. **Nhiễu vĩ mô đồng pha (Market Beta):** Xu hướng chung của toàn thị trường che lấp biến động nội tại (alpha) của từng tài sản đơn lẻ.
3. **Rò rỉ chồng chéo nhãn (Target Overlap Leakage):** Nhãn $y_t$ là lợi suất tích lũy từ $t$ đến $t + H$. Các nhãn gần biên tập huấn luyện sẽ giao thoa sang tập kiểm định nếu chỉ chia cắt thời gian cố định.

---

## Kiến trúc Giải pháp

```text
┌-----------------------------------------------------------------------------------------┐
| 1. THU NẠP DỮ LIỆU & TIỀN XỬ LÝ BỘ NHỚ                                                  |
| - Dữ liệu thô: train.parquet (5.3M dòng) & test.parquet (1.4M dòng)                     |
| - Ép kiểu tối ưu: float64 -> float32 | int64 -> int32 (giảm 50% RAM)                    |
└-----------------------------------------------------------------------------------------┘
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 2. PHÂN RÃ THEO 4 HORIZONS ĐỘC LẬP                                                      |
| - Phân rã luồng riêng biệt: Horizon H in {1, 3, 10, 25}                                 |
| - Mỗi Horizon huấn luyện mô hình và xây dựng cấu trúc nhãn riêng                        |
└-----------------------------------------------------------------------------------------┘
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 3. HỆ THỐNG TRÍCH XUẤT ĐẶC TRƯNG HYBRID                                                 |
| - Khai thác vi pha, động lực học bậc hai và triệt tiêu nhiễu thị trường                 |
└-----------------------------------------------------------------------------------------┘
                                             |
              +------------------------------+------------------------------+
              | Tín hiệu ngành               | Bóc tách vi pha              | Động học & Z-Score
              ▼                              ▼                              ▼
┌---------------------------┐  ┌---------------------------┐  ┌---------------------------┐
|        STATIC PROXY       |  |    PSEUDO-WAVELET (EMA)   |  |     ĐỘNG HỌC & Z-SCORE    |
| - Target: feature_al      |  | - Ngắn : EMA(3) - EMA(10) |  | - ROC & Gia tốc bậc 2     |
| - Nhóm sub_category,      |  | - Trung: EMA(10)-EMA(25)  |  | - Proxy Spread (ngành)    |
|   sub_code                |  | - Dài  : EMA(25)-EMA(50)  |  | - Cross-Sectional Z-Score |
| - Khóa ts_index <= 3500   |  | - Bóc tách vi pha nhịp    |  |   tại từng timestamp      |
|   (chống look-ahead)      |  | - Khử độ trễ đệ quy       |  | - Cờ ngoại lệ (|z| > 3.0) |
└---------------------------┘  └---------------------------┘  └---------------------------┘
              |                              |                              |
              +------------------------------+------------------------------+
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 4. DYNAMIC EMBARGO VALIDATION SPLIT                                                     |
| - Vùng đệm cách ly: delta = H + 25 bước (chống Target Overlap Leakage)                  |
| - Tập Huấn luyện: ts_index <= 3500 - delta | Tập Kiểm định: ts_index > 3500             |
└-----------------------------------------------------------------------------------------┘
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 5. HUẤN LUYỆN 5-SEED LIGHTGBM ENSEMBLE                                                  |
| - Khởi tạo 5 hạt giống: Seeds [42, 2024, 12345, 99, 420]                                |
| - Khôi phục trọng số gốc (loại bỏ Time Decay 40x gây quá khớp)                          |
| - Tối ưu trực tiếp hàm mục tiêu Weighted RMSE (W-RMSE)                                  |
└-----------------------------------------------------------------------------------------┘
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 6. HỆ THỐNG GIÁM SÁT PHƯƠNG SAI TỰ ĐỘNG                                                 |
| - Kiểm tra tỷ lệ triệt tiêu (zero-rate audit với sai số atol = 1e-12)                   |
| - Cảnh báo suy thoái phương sai nếu tỷ lệ dự báo 0 vượt ngưỡng 60%                      |
└-----------------------------------------------------------------------------------------┘
                                             |
                                             ▼
┌-----------------------------------------------------------------------------------------┐
| 7. SEQUENTIAL INFERENCE & XUẤT SUBMISSION                                               |
| - Nối buffer 50 bước train_tail vào test set để bảo toàn Lag và EMA                     |
| - Dự báo tuần tự, xuất submission.csv (1.44M dòng | Private: 0.2475)                    |
└-----------------------------------------------------------------------------------------┘
```

### Các Kỹ thuật Cốt lõi:
- **Dynamic Embargo Split ($\delta = H + 25$):** Ép lùi ranh giới tập huấn luyện một khoảng $H + 25$ bước, triệt tiêu hoàn toàn hiện tượng Target Overlap Leakage giữa train và validation.
- **Static Proxy Target Encoding:** Tính trước giá trị trung bình của `feature_al` (tương quan tuyến tính `r > 0.94` với target) theo `sub_category` và `sub_code` trên các mẫu `ts_index <= 3500`, loại bỏ forward look-ahead bias.
- **Bộ lọc Pseudo-Wavelet:** Bóc tách vi pha và xu hướng đa chu kỳ qua chênh lệch các đường EMA (EMA3 - EMA10, EMA10 - EMA25, EMA25 - EMA50).
- **Gia tốc Động học (2nd-Order Acceleration):** Đạo hàm bậc hai cắt trục 0 trước khi vận tốc tạo đỉnh, giúp cây quyết định phân nhánh đón đầu tại điểm uốn cấu trúc.
- **Chuẩn hóa Cross-Sectional Z-Score:** Chuẩn hóa theo từng mốc thời gian để loại bỏ biến động vĩ mô toàn thị trường, giữ lại sức mạnh alpha riêng biệt của tài sản.
- **Khôi phục Trọng số Gốc:** Gỡ bỏ hoàn toàn hàm phạt Time Decay 40x, trả lại hàm mục tiêu tự nhiên giúp mô hình duy trì tính tổng quát hóa trên tập kiểm thử thực tế.

---

## Cấu trúc Thư mục

```
Hedge_Fund/
├── config/
│   └── config.yaml              # Cấu hình siêu tham số mô hình, seeds, horizons
├── src/
│   ├── data/
│   │   ├── loader.py            # Đọc Parquet và ép kiểu tối ưu RAM
│   │   └── proxy_encoder.py     # Proxy Target Encoder chống rò rỉ dữ liệu
│   ├── features/
│   │   └── engineering.py       # Wavelets, gia tốc, cross-sectional z-score
│   ├── models/
│   │   ├── metrics.py           # Weighted RMSE và kiểm soát tỷ lệ zero-rate
│   │   ├── trainer.py           # Trainer LightGBM với Dynamic Embargo
│   │   └── inference.py         # Sequential inference ghép nối train_tail
│   └── utils/
│       └── helpers.py           # Seeding, logging
├── notebooks/
│   └── walkthrough_demo.ipynb   # Demo pipeline chạy trên dữ liệu mô phỏng (~10s)
├── scripts/
│   ├── train.py                 # Script huấn luyện và suy luận từ terminal
│   └── evaluate.py              # Script kiểm tra phân phối file submission.csv
├── tests/
│   └── test_pipeline.py         # Bộ kiểm thử unit tests kiểm tra logic cốt lõi
└── requirements.txt             # Danh sách thư viện phụ thuộc
```

---

## Hướng dẫn Chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy kiểm thử Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Chạy Demo nhanh (không cần tải dữ liệu 900MB)
Mở `notebooks/walkthrough_demo.ipynb` để chạy thử toàn bộ pipeline trên dữ liệu thị trường mô phỏng trong khoảng 10 giây.

### 4. Huấn luyện toàn bộ Pipeline
Đặt 2 file `train.parquet` và `test.parquet` vào thư mục `data/raw/` và thực thi:
```bash
python scripts/train.py --config config/config.yaml
```
Kiểm tra phân phối file submission:
```bash
python scripts/evaluate.py --submission submission.csv
```

---

## Nhóm tác giả

- **Lê Văn Hiệp**
- **Nguyễn Duy Hiếu**
- **Vũ Minh Hiếu**
- **Trần Trung Hiếu**
