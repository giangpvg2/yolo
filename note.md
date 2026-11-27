# Sổ tay bàn giao & Hướng dẫn hành động sau khi chạy Colab (`note.md`)

Tài liệu này tóm tắt toàn bộ ngữ cảnh dự án, cách chuyển sang máy mới, cách khôi phục phiên chạy trên Google Colab, và **quy trình đọc kết quả để làm tiếp Giai đoạn 4–5** theo đúng `PLAN.md`.

---

## 1. Chuyển dự án sang máy mới (Setup nhanh)

### 1.1. Lấy mã nguồn (ưu tiên qua GitHub thay vì copy thủ công cả thư mục nặng)
Mã nguồn sạch (đã có đủ 47/47 unit tests và notebook Colab) đã được đẩy lên GitHub:
```bash
git clone https://github.com/giangpvg2/yolo.git YOLOv8_with_P2
cd YOLOv8_with_P2
```
> **Nếu copy thư mục bằng USB/ổ cứng:** Không cần copy thư mục `.venv/` (môi trường ảo gắn cứng đường dẫn máy cũ sẽ lỗi trên máy mới). Hãy xóa `.venv/` cũ nếu lỡ copy sang và tạo mới theo mục 1.3.

### 1.2. Những dữ liệu/kết quả KHÔNG nằm trên GitHub (cần tải từ Google Drive về)
Do `.gitignore` chặn các file nặng, khi chạy xong trên Google Colab bạn tải thư mục `MyDrive/YOLOv8_with_P2_workspace/` về đặt vào gốc dự án theo cấu trúc:
1. **`reports/`** *(Bắt buộc tải về toàn bộ — rất nhẹ)*:
   - `reports/data_decisions.md`: Nhật ký tách ảnh trùng `train`/`val` (Cổng `D0`).
   - `reports/audit-dev/`: Chứa `groups_val.json`, `groups_test.json`, `duplicates.json`, `summary.json`.
   - `reports/eval-B0-s0/`, `reports/eval-B1-s0/`, `reports/eval-B0E-s0/`: Toàn bộ bảng điểm COCO tổng, từng lớp, theo mức độ mờ/rối nền và đường cong `froc.csv`.
2. **`runs/detect/`** *(Tải chọn lọc các file quan trọng của `dev-B0-s0`, `dev-B1-s0`, `dev-B0E-s0`)*:
   - `experiment.json`, `training_telemetry.json`, `results.csv`, `args.yaml`.
   - **`weights/best.pt`** *(Bắt buộc giữ lại trọng số này để nạp vào bước Chẩn đoán Giai đoạn 4)*.
3. **`URPC2020.zip`** *(Tải về giải nén vào `archive/URPC2020/` nếu muốn chạy script chẩn đoán cục bộ trên máy mới)*.

### 1.3. Khởi tạo lại môi trường Python trên máy mới
Code trong `train.py` khóa cứng phiên bản thư viện để đảm bảo tái lập khoa học (`ultralytics==8.3.232`, `torch==2.9.1`, `torchvision==0.24.1`, `pycocotools==2.0.11`):
```bash
python3 -m venv .venv
source .venv/bin/activate

# Nếu máy mới CHỈ dùng CPU để đọc kết quả / viết code:
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# (Hoặc nếu máy mới có GPU NVIDIA CUDA 12.6 / 12.8):
# python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q   # Kỳ vọng: 47 passed
```

---

## 2. Bức tranh nghiên cứu cốt lõi (Để nhớ nhanh & trao đổi với GVHD)

### 2.1. Lộ trình 8 giai đoạn & Vị trí hiện tại
`Khóa bài toán & dữ liệu (D0)` → `Kiểm tra GPU (D1)` → **`Chạy 3 Baseline (D2 — Đang chạy trên Colab)`** → `Chẩn đoán lỗi & chốt 1 nhánh H-A hoặc H-F (D3)` → `Code 1 Prototype thắng đối chứng (D4)` → `Đánh giá đa seed 0,1,2 (D5)` → `Viết báo cáo/paper`.

- **Trạng thái code hiện tại:**
  - Phần **Baseline (`D0` → `D2`)**: **Đã code xong 100%** (`train.py`, `eval.py`, `dataset_audit.py`, `prepare_dev_split.py`, `clahe.py`, `colab_baseline.ipynb`).
  - Phần **Chẩn đoán (`D3`) & Thuật toán mới (`D4`)**: **Chưa viết code (0%)** — bắt buộc đợi kết quả Baseline để biết lỗi thật nằm ở đâu rồi mới code đúng 1 nhánh.

### 2.2. Phân biệt dễ hiểu 2 giả thuyết nghiên cứu (`H-A` vs `H-F`)

| Tiêu chí | Nhánh `H-A`: Ổn định gán nhãn (*Assignment*) — **Ưu tiên số 1** | Nhánh `H-F`: Lọc đặc trưng/tần số (*Feature*) — **Dự phòng** |
|---|---|---|
| **Bản chất lỗi** | Ảnh mờ/lệch màu làm bộ chọn mẫu dương `TAL` của YOLOv8 bị nhảy chập chờn giữa các ô lưới khi học vật thể nhỏ. | Nhánh độ phân giải cao `P2` phóng rõ vật thể nhỏ nhưng truyền luôn cả nhiễu cặn nước, sỏi đá (`clutter`) gây báo nhầm. |
| **Triệu chứng ở Baseline** | Tỷ lệ bỏ sót `FN_small_rate` cao vọt ở nhóm ảnh suy giảm nặng (`severity = Severe`). | Báo nhầm `FP_per_image` tăng vọt ở `B1-Raw` trên nền sỏi đá rối (`clutter = Hard Clutter`). |
| **Cách sửa** | Huấn luyện 2 view cùng tọa độ (`A`: rõ hơn, `B`: suy giảm); mượn mẫu dương tốt từ `A` dẫn đường **có điều kiện** cho `B` (vẫn dùng nhãn `GT` thật). | Tách tần số (ví dụ `Haar wavelet`) ở tầng `P2` + ngữ cảnh tầng `P3` làm cổng lọc (`gate`) dập nhiễu nền. |
| **Tốc độ lúc chạy thật (`Inference`)** | **Giữ nguyên 100%** (bỏ toàn bộ nhánh phụ khi xuất mô hình). | Chậm hơn nhẹ (phải tính thêm bộ lọc/gate trong thân mạng). |
| **Tính mới & Rủi ro trùng** | **Dễ tạo điểm mới hơn:** Khác `RFLA`/`SimD`/`STAL` (chỉ xét hình học 1 ảnh) và khác `consistency` mù quáng (`E4`/`E5`). Cần đọc kỹ `Consistent-Teacher` và `DCP-UDet` để vạch ranh giới. | **Rủi ro đụng hàng cao:** Đã có rất nhiều bài làm `P2` + wavelet/attention (`SET`, `WTConv`, `ASFF`, `DM-EFS`, `DSG-YOLO26`). |

### 2.3. Cách trao đổi với Giảng viên hướng dẫn (GVHD) về "Tính mới"
- **Quy tắc:** Không hỏi chung chung *"làm gì cho mới?"* và không nghĩ từ *"gắn thêm block X vào YOLO"*. Đi từ công thức: **Đo tại sao công cụ `X` (`TAL` / `P2`) thất bại ở điều kiện `Z` (ảnh suy giảm dưới nước) → Sửa đúng điểm nghẽn `Z` với chi phí suy luận tối thiểu**.
- **Câu hỏi trọng tâm dành cho GVHD sau khi có số Baseline:**
  1. Với mục tiêu đầu ra của đề tài, thầy/cô ưu tiên đóng góp dạng **chiến lược huấn luyện/gán nhãn không làm chậm tốc độ chạy thực tế (`H-A`)** hay thiết kế module lọc đặc trưng trong mạng (`H-F`)?
  2. Nếu thực nghiệm chẩn đoán xác nhận hiện tượng nhảy mẫu dương dưới suy giảm ảnh là có thật và cơ chế chuyển mẫu dương có điều kiện (`E6`) vượt qua các đối chứng (`E3` hai view độc lập, `E4` consistency, `E5` chép thẳng, `E7` mở rộng độ phủ kiểu `STAL`), thì mức đóng góp đó đã đáp ứng kỳ vọng của thầy/cô chưa?

---

## 3. Vận hành & Khôi phục phiên chạy trên Google Colab

- **Link mở trực tiếp notebook trên Colab:**
  https://colab.research.google.com/github/giangpvg2/yolo/blob/main/colab_baseline.ipynb
- **Vị trí đặt dữ liệu trên Google Drive:** `MyDrive/YOLOv8_with_P2_workspace/URPC2020.zip` (hoặc `MyDrive/URPC2020.zip`).
- **3 kịch bản chạy chính (`100 epochs`, `seed 0`, `640 px`, `batch 16`):**
  1. `dev-B0-s0` (`B0-Raw`): `YOLOv8n` trên ảnh gốc đã tách rò rỉ.
  2. `dev-B1-s0` (`B1-Raw`): `YOLOv8n-P2` trên ảnh gốc đã tách rò rỉ.
  3. `dev-B0E-s0` (`B0-Enhanced`): `YOLOv8n` trên ảnh `Gray World + CLAHE`.
  *(Lưu ý về `batch`: giữ `batch=16` để đủ ~34.600 bước cập nhật khi train từ đầu và chừa VRAM cho Giai đoạn 5 chạy 2 view; nếu đổi sang `32` trên A100/L4 thì bắt buộc chạy lại đồng nhất cả 3 kịch bản ở `32`).*

### Cách chạy tiếp (`--resume`) khi tắt máy / đứt kết nối giữa chừng
Nếu một kịch bản đang chạy dở (ví dụ `dev-B1-s0` dừng ở epoch 30), **không chạy lại từ đầu**:
1. Mở lại Colab → Chạy **Ô 1** (Mount Drive + Clone code) → **Ô 2** (Cài thư viện) → **Ô 3 (Cổng D0)** (Giải nén lại ảnh ra `/content/datasets/URPC2020_dev`).
2. Bỏ qua Ô Cổng D1 và Ô Cổng D2 cũ.
3. Chạy tại **Ô Resume**:
   ```python
   !python train.py --resume "{PERSISTENT_RUNS}/dev-B1-s0/weights/last.pt" --device 0
   !python eval.py --data configs/data_dev.yaml --weights "{PERSISTENT_RUNS}/dev-B1-s0/weights/best.pt" --groups "{PERSISTENT_REPORTS}/audit-dev/groups_val.json" --output "{PERSISTENT_REPORTS}/eval-B1-s0" --device 0
   ```
4. Chạy tiếp ô `B0-Enhanced` và ô tổng hợp bảng ở cuối.

---

## 4. Khi có kết quả trên Colab thì đọc file nào và quyết định ra sao?

### 4.1. Năm chỗ cần mở ra đối chiếu giữa `eval-B0-s0`, `eval-B1-s0` và `eval-B0E-s0`
1. **Hiệu quả vật thể nhỏ (`metrics.csv`, dòng `group = all`):** So `AP_small` (metric chính), `AR_small`, `FN_small_rate`, và kiểm tra `AP`, `AP_medium`, `AP_large` không bị tụt bất thường.
2. **Đánh đổi báo nhầm công bằng (`metrics.csv` & `froc.csv`):** So `FP_per_image` ở ngưỡng `0.25`, đồng thời mở `froc.csv` so **`FP_per_image` tại cùng một mức `recall_small`** (ví dụ cùng `recall_small = 0.50`).
3. **Chi tiết từng loài (`metrics_per_class.csv`):** Xem `AP_small` và `FN_small_rate` trên từng lớp (`holothurian`, `echinus`, `scallop`, `starfish`).
4. **Điểm yếu theo điều kiện nước & nền (`metrics.csv`, các dòng nhóm):**
   - Nhóm `severity/.../Clear`, `Moderate`, `Severe`: Mờ nặng (`Severe`) làm `FN_small_rate` tăng mạnh cỡ nào?
   - Nhóm `clutter/.../Smooth Seabed`, `Hard Clutter`: Nền rối (`Hard Clutter`) có làm `FP_per_image` của `B1-Raw` vọt cao không?
5. **Độ ổn định & Chi phí phần cứng (`training_telemetry.json` & `results.csv`):**
   - Kiểm tra `finite_metrics: true` (không dính `NaN`/`Inf`), đường mAP cuối epoch 100 đã hội tụ phẳng chưa.
   - Ghi nhận `total_training_seconds`, `seconds_per_epoch` và `max_peak_vram_allocated_mb` để lập ngân sách GPU cho Giai đoạn 4–6.

### 4.2. Bảng quyết định hành động theo kết quả thực tế (Từ Cổng `D2` sang Giai đoạn 4)

| Kịch bản số liệu thu được | Chọn mô hình nền | Việc phải làm ngay tiếp theo (Giai đoạn 4 — Cổng `D3`) |
|---|---|---|
| **1. `B1-Raw` tăng rõ `AP_small`, nhưng nhóm `Severe` vẫn bỏ sót nhiều** | **`B1-Raw` (`YOLOv8n-P2`)** | Lấy `dev-B1-s0/weights/best.pt` viết **logger chẩn đoán nhánh `H-A`** (sinh cặp ảnh `A/B`, đo độ nhảy mẫu dương `TAL` trước NMS). |
| **2. `B1-Raw` tăng `AR_small` nhưng `FP_per_image` ở `Hard Clutter` vọt mạnh** | **`B1-Raw` (`YOLOv8n-P2`)** | 1) Kiểm tra mắt ~50 hộp `FP` mới của `B1` xem có phải do bộ gốc thiếu nhãn không.<br>2) Đo **`H-A`** trước; nếu `H-A` yếu thì mở chẩn đoán tần số **`H-F`** (Giai đoạn 4.2). |
| **3. `B1-Raw` ≤ `B0-Raw` dù tốn thêm nhiều VRAM/thời gian** | **`B0-Raw` (`YOLOv8n`)** | Bỏ nhánh `P2` và bỏ `H-F`; chuyển toàn bộ khảo sát **`H-A` sang nền `B0-Raw`** (theo mục 9.1 của `PLAN.md`). |
| **4. `B0-Enhanced` (`CLAHE`) thua `B0-Raw`** *(hoặc chỉ thắng ở `Severe`)* | Giữ nhánh `Raw` làm chính | Nếu thua: dùng làm đối chứng âm chứng minh tiền xử lý cứng gây hại. Nếu thắng mạnh ở `Severe`: cân nhắc dùng ảnh cân bằng màu làm view tham chiếu `A` khi train `H-A`. |
| **5. Chênh lệch `B0` và `B1` quá sít sao (< 0.3–0.5 AP) hoặc chưa hội tụ** | Chưa chốt vội | Kiểm tra `results.csv`; chạy thêm **`seed 1` và `seed 2`** cho `B0`/`B1` trước khi chốt nền. |

---

## 5. Checklist các bước thực hiện trên máy mới ngay khi có kết quả Colab

- [ ] **Bước 1:** Tải `reports/` và `runs/detect/*/weights/best.pt` + `training_telemetry.json` + `results.csv` từ Google Drive về máy mới.
- [ ] **Bước 2:** Duyệt mẫu thủ công trong `reports/audit-dev/review_sample.csv` (và kiểm tra 17 cặp ảnh gần trùng trong `duplicates.json`), cập nhật `reviewed: true` trong `groups_val.json` nếu cần chấm lại nhóm đã duyệt.
- [ ] **Bước 3:** Hoàn thiện `reports/problem_statement.md` (1 trang chốt lỗi mục tiêu, metric chính `AP_small` và bảng đối chiếu các bài gần nhất: `TAL`, `STAL`, `Consistent-Teacher`, `DCP-UDet`).
- [ ] **Bước 4:** Gửi bảng so sánh 3 Baseline (`B0-Raw`, `B1-Raw`, `B0-Enhanced`) cho GVHD để xác nhận hướng đi vào Giai đoạn 4.
- [ ] **Bước 5 (Bắt đầu code Giai đoạn 4 — `D3`):** Viết công cụ trích xuất dự đoán dày trước NMS (`dense predictions`) và `assignment logger` cho cặp ảnh `A` (gốc) / `B` (làm mờ, giảm tương phản, lệch màu nhẹ) trên checkpoint `best.pt` cố định để đo `positive count`, `unassigned GT rate` và `assignment churn`.
- [ ] **Bước 6 (Chỉ làm sau khi `D3` có tín hiệu):** Triển khai tuần tự prototype `E3` (2 view `TAL` độc lập) → `E5` (chép thẳng `A → B`) → `E6` (chuyển có điều kiện theo bằng chứng + fallback `TAL`) và các đối chứng `E4`/`E7`.

---

### Prompt gợi ý khi mở trợ lý AI trên máy mới
Khi chuyển sang máy mới và đã chép thư mục `reports/` từ Colab về, bạn chỉ cần nhắn câu sau để AI nắm lại toàn bộ mạch việc ngay lập tức:
> *"Hãy đọc `PLAN.md`, `note.md`, `docs/research/assignment-roadmap.md` và các kết quả thực tế trong `reports/eval-B0-s0/`, `reports/eval-B1-s0/`, `reports/eval-B0E-s0/`. Phân tích giúp tôi kết quả Cổng D2, xác nhận kịch bản đang xảy ra và đề xuất thiết kế chi tiết để bắt đầu code bước chẩn đoán Giai đoạn 4 (Cổng D3)."*
