# YOLOv8-P2: baseline, evaluation và dataset audit

Pipeline nghiên cứu phát hiện vật thể nhỏ dưới nước. Bản hiện tại triển khai
chuẩn hóa train/eval và audit dữ liệu; chưa triển khai wavelet, M1–M4 hoặc
assignment hai view. Xem [thiết kế và protocol](docs/pipeline.md).

## Cài đặt

Python 3.11 trở lên. Tạo môi trường riêng, chọn một backend PyTorch rồi cài
các dependency đã khóa:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Trên Colab/GPU, thay lệnh cài backend CPU bằng cặp CUDA tương thích với driver;
ví dụ CUDA 12.8:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Các backend và cặp phiên bản được liệt kê tại
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
Môi trường CPU của dự án đã được kiểm tra với Python 3.13.9. Recipe dùng
Ultralytics 8.3.232 và chặn train khi Torch/Torchvision khác phiên bản đã khóa
hoặc thiếu compiled operators. CUDA build suffix như `+cu128` được chấp nhận.
Không cài CPU backend vào phiên GPU dự định dùng để train.

## Chuẩn bị dữ liệu

**Dữ liệu hiện được chọn:** `archive/URPC2020/`. Dùng trực tiếp với
`configs/data_urpc_raw.yaml`; không cần chạy script di chuyển/giải nén bên dưới.
Source có 5543 ảnh train, 1200 ảnh valid và 800 ảnh test. Class IDs giữ theo
YAML nguồn: `0=holothurian, 1=echinus, 2=scallop, 3=starfish`.
`archive/URPC2020/data.yaml` đã được sửa đường dẫn portable; bản gốc được
sao lưu ở `reports/urpc-source-repair/data.original.yaml`.

```bash
python train.py --baseline B0-Raw --dry-run
python dataset_audit.py --data configs/data_urpc_raw.yaml --splits train val test --output reports/urpc-source-audit-new
```

Kết quả kiểm tra nguồn xem `docs/research/urpc-data-check.md`. Các bước tạo
`URPC2020_YOLO` bên dưới là luồng chuẩn bị thay thế cho archive layout tương
ứng; không cần chạy chúng cho source đã chọn.

Để loại bỏ rò rỉ ảnh trùng chính xác giữa `train` và `val`/`test` (Cổng D0) mà
không làm thay đổi tập `val`/`test` hay sửa trực tiếp thư mục nguồn, tạo bản
phát triển có khóa manifest:

```bash
python preprocessing_data/prepare_dev_split.py \
  --data configs/data_urpc_raw.yaml \
  --output datasets/URPC2020_dev \
  --yaml-output configs/data_dev.yaml \
  --report reports/data_decisions.md
```

Trên Google Colab, có thể thêm `--link-mode hardlink` để tạo tức thời không tốn
thêm dung lượng đĩa, hoặc truyền `--exclude-train-file` sau khi review các cặp
ảnh gần trùng. Xem quy trình trọn gói trên Colab tại `colab_baseline.ipynb`.

Đặt `URPC2020.zip` ở root dự án. Archive cần có
`URPC2020/{train,valid,test}/{images,labels}`. Chạy từ root:

```bash
bash preprocessing_data/scripts.sh
```

Script sao chép dữ liệu sang `URPC2020_YOLO/{images,labels}/{train,val,test}`,
giữ nguyên nguồn và từ chối ghi đè thư mục đích đã tồn tại. Nếu dataset đã theo
chuẩn YOLO, cấu hình phải trỏ đúng root/layout của bản đó.

Data YAML giải `path` tương đối với vị trí file YAML, sau đó giải các split
tương đối với `path`. Pipeline hiện hỗ trợ split là thư mục ảnh, bao gồm thư mục
con; chưa hỗ trợ split dạng danh sách hoặc file TXT. Nhãn là YOLO detection
`class xc yc w h` chuẩn hóa. Phải kiểm tra thứ tự lớp với nguồn dữ liệu:
`holothurian, echinus, scallop, starfish` cho source hiện được chọn.

Ảnh âm có file nhãn rỗng; thiếu file nhãn mặc định là lỗi. Audit/eval có
`--allow-missing-labels` cho trường hợp người dùng chủ động coi thiếu nhãn là
background. Không dùng tùy chọn này để bỏ qua lỗi dữ liệu chưa xác minh.

Tạo nhánh ảnh tăng cường để đối chứng (hỗ trợ trực tiếp cả `--data` YAML lẫn
thư mục `--input`):

```bash
python preprocessing_data/clahe.py --data configs/data_dev.yaml --output datasets/URPC2020_dev_clahe --yaml-output configs/data_dev_clahe.yaml
```

Script này chạy **Gray World + CLAHE**, giữ nguyên labels/split, hỗ trợ ảnh trong
thư mục con và từ chối ghi đè output. Nếu chỉ có train/val, thêm
`--splits train val`. Vì có hai phép xử lý, baseline mang tên `B0-Enhanced`.
Các file `configs/data_urpc_raw.yaml` và `configs/data_urpc_clahe.yaml` đã có sẵn.

Với dataset khác, có thể sinh YAML portable và khai báo lại lớp:

```bash
python preprocessing_data/create_yaml.py --dataset datasets/custom --output configs/custom.yaml --names class_a class_b
```

## Audit trước khi train

Nguồn hiện tại đã được kiểm tra/sửa bằng công cụ có backup từng file:

```bash
python preprocessing_data/repair_labels.py --data configs/data_urpc_raw.yaml --output reports/repair-preview
```

Mặc định chỉ lập báo cáo; thêm `--apply` mới ghi nhãn sau khi sao lưu. Công cụ
clip hộp vượt biên về khung ảnh và bỏ hộp diện tích 0, giữ nguyên class ID;
dữ liệu sai class/center hoặc size âm sẽ yêu cầu kiểm tra thủ công.
Nhật ký lượt sửa ở `reports/urpc-source-repair/labels/repair.json`, backup
nhãn ở `reports/urpc-source-repair/labels/original_labels/`.

Export cũ `URPC.v1i.yolov8` (nếu còn giữ): metadata khai báo **5 lớp**
`Fish, Scallop, Seacucumber, Seaurchin, Star-fish`, chỉ có train/valid.
`configs/data_urpc_local.yaml` giữ đúng class ID và sửa cách giải đường dẫn cho
export này. Không dùng cấu hình URPC2020 bốn lớp cho nó; cần xác minh nguồn và
split trước khi gọi đây là benchmark URPC2020. Không dùng cấu hình này cho
source hiện tại. Các lệnh dưới đây chỉ dành cho export cũ:

```bash
python dataset_audit.py --data configs/data_urpc_local.yaml --splits train val --output reports/urpc-local
python train.py --baseline B0-Raw --data configs/data_urpc_local.yaml --name local-B0-s0 --device 0
python train.py --baseline B1-Raw --data configs/data_urpc_local.yaml --name local-B1-s0 --device 0
```

Các cấu hình raw/CLAHE bốn lớp ở phần trên dành cho dataset được chuẩn bị theo
quy trình URPC2020; chúng không tự chuyển đổi class ID của export năm lớp.

```bash
python dataset_audit.py --data configs/data_urpc_raw.yaml --splits train val --output reports/audit
```

Các đầu ra:

Nếu nhãn/ảnh không hợp lệ, audit gom các ảnh lỗi vào `validation_issues.json`,
ghi báo cáo lỗi và thoát với mã khác 0. Không xuất thống kê/nhóm trên một tập
con âm thầm bỏ ảnh lỗi. Nhãn nguồn được giữ nguyên; train/eval vẫn kiểm tra
nghiêm ngặt. Khi dữ liệu hợp lệ, các đầu ra gồm:

Sai số export vượt biên tối đa 0,01 pixel được chấp nhận và thống kê trong
`border_rounding_boxes`; tọa độ nguồn không bị cắt hay sửa.

- `dataset_audit.md`, `summary.json`: số ảnh/hộp theo lớp, quantile tỷ lệ diện
  tích và số lượng các nhóm, bao gồm giao giữa severity và clutter.
- `manifest.csv`: đường dẫn tương đối, kích thước, SHA-256 ảnh/nhãn, dHash.
- `objects.csv`: diện tích pixel ở ảnh gốc và tỷ lệ diện tích của từng hộp.
- `duplicates.json`: ảnh trùng chính xác, cờ trùng split và tối đa 1000 cặp gần
  trùng giữa các split cần kiểm tra mắt.
- `groups_val.json` (và file tương ứng các split): nhóm severity/clutter đề xuất.
- `review_sample.csv`: tối đa 300 ảnh mỗi split val/test, lấy mẫu theo nhóm.
- `ground_truth_val.json` (và các split khác): COCO GT để xác định ID ảnh/lớp.

Severity là proxy kết hợp tương phản cục bộ, độ sắc nét và mất cân bằng màu.
Clutter dùng mật độ biên ngoài các hộp GT; nền quá ít được gán `Unknown`.
Các chỉ số này không đo trực tiếp độ đục nước. Ngưỡng cố định nằm trong
`configs/audit.yaml`; kiểm tra trên development set rồi khóa trước khi dùng test.

Kiểm tra ảnh trong sample; sửa nhãn nhóm trong `groups_<split>.json` và đặt
`reviewed: true` cho ảnh đã kiểm tra. Không sửa fingerprint. CSV sample là danh
sách hỗ trợ xem ảnh, không tự nhập ngược vào JSON. Hash chỉ tìm ứng viên trùng;
không chứng minh hai dataset độc lập nguồn cảnh quay.

## Train baseline

```bash
python train.py --dry-run
python train.py --baseline B0-Raw --device 0
python train.py --baseline B0-Enhanced --device 0
python train.py --baseline B1-Raw --device 0
```

B0 dùng `yolov8n.yaml`; B1 dùng `yolov8n-p2.yaml` đóng gói cùng Ultralytics.
Không cần file `yolov8n_p2.yaml` tự tạo. Mặc định cả ba train **từ đầu**,
640 px, 100 epochs, batch 16, seed 0 và recipe dùng chung ở `configs/train.yaml`.
AMP tắt để tránh bước kiểm tra AMP tự tải checkpoint tham chiếu.

Dùng `--seed 1`, `--seed 2` cho nhiều seed. Có thể đổi `--data`, `--config`,
`--epochs`, `--batch`, `--imgsz`, nhưng phải áp dụng cùng protocol cho các
baseline so sánh. Đặt `--device cpu` để kiểm tra trên CPU. Dry-run chỉ in cấu
hình và tình trạng thư mục dữ liệu, không xác nhận dữ liệu hợp lệ hay khả năng
train. Model không tải pretrained weights; Ultralytics có thể tải font cho plotting.

Ví dụ kết quả B1 ở `runs/detect/B1-Raw_s0_640/`:

- `experiment.json`: recipe, phiên bản môi trường, `code_fingerprint`, fingerprint train/val và lịch sử phiên chạy.
- `training_telemetry.json`: thời gian train, giây/epoch, tên GPU, `peak_vram_allocated_mb` và cờ kiểm tra `NaN`/`Inf` (`finite_metrics`).
- `args.yaml`: tham số Ultralytics đã giải đầy đủ.
- `weights/best.pt`, `weights/last.pt`: checkpoint.
- `evaluation/metrics.csv`, `evaluation/metrics_per_class.csv`, JSON metrics/GT/predictions (gồm cả kiểm tra cảnh đông `crowding` tại `maxDets=300`) và lỗi theo ảnh:
  tự động chấm checkpoint tốt nhất trên validation sau train.

Thư mục run đã tồn tại sẽ bị từ chối; dùng `--name` khác cho lượt mới. Resume
một lượt đang dở bằng checkpoint còn optimizer state:

```bash
python train.py --resume runs/detect/B1-Raw_s0_640/weights/last.pt --device 0
```

Resume dùng recipe/checkpoint của lượt gốc, không kết hợp với override recipe.
Đường dẫn dữ liệu cần còn hợp lệ. Checkpoint đã hoàn tất có thể bị Ultralytics
strip optimizer và không dùng để resume. `--skip-eval` bỏ chấm COCO tự động,
không tắt validation của trainer. Resume không đảm bảo kết quả bitwise giống
lượt chạy liên tục.

## Đánh giá độc lập và theo nhóm

```bash
python eval.py --data configs/data_urpc_raw.yaml --weights runs/detect/B1-Raw_s0_640/weights/best.pt --groups reports/audit/groups_val.json --output reports/B1-val --device 0
```

Có `AP_small`, `AR_small`, AP/AR tổng, `FP_per_image`, `FN_small` và
`FN_small_rate`. AP/AR nằm trong [0,1]; metric không xác định được ghi null/ô
CSV trống. COCO dùng diện tích bbox ở **ảnh gốc**, mốc small 1024 pixel²,
IoU 0.50–0.95, maxDets 1/10/100. Không gọi tỷ lệ diện tích chuẩn hóa là COCO small.

FP/FN dùng confidence 0.25 và matching một-một theo lớp tại IoU 0.5.
Thay bằng `--confidence` và `--match-iou`; hai ngưỡng này không đổi phép tính AP.
NMS có `--nms-iou` riêng. Predict giữ score từ 0.001, cap mặc định 300,
đọc ảnh tuần tự từ đĩa. Báo cáo protocol ghi rõ các ngưỡng và cap.

`froc.csv` ghi recall, recall_small và FP/image theo confidence trên toàn split,
giúp so sánh ở cùng recall. Chỉ dùng development data để chọn điểm vận hành;
khóa cách chọn ngưỡng trước khi áp dụng lên test.

Group manifest phải khớp nội dung ảnh, nhãn, kích thước và split hiện tại.
Các hàng `severity/proposed/...` khác với `severity/reviewed/...`; tương tự
với clutter. Các hàng nhóm chỉ đánh giá trên ảnh thuộc nhóm tương ứng.

Chấm lại predictions đã xuất mà không chạy detector:

```bash
python eval.py --data configs/data_urpc_raw.yaml --predictions reports/B1-val/predictions.json --output reports/B1-replay
```

Mặc định dùng file `ground_truth.json` bên cạnh predictions để kiểm tra ID và
nội dung dataset. Predictions từ hệ thống khác phải dùng đúng ID trong GT và
truyền `--reference-gt reports/audit/ground_truth_val.json`.

Khi protocol đã khóa, dùng `--split test` với group manifest của test nếu có.
Không dùng test để chọn checkpoint, ngưỡng hay module. Mỗi thư mục output eval
và audit phải mới hoặc rỗng; kết quả cũ được giữ nguyên.

## Kiểm tra

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Các kiểm thử dùng dữ liệu giả, kiểm tra COCO perfect/empty cases, FP sai lớp/
trùng, FN_small, nhãn lỗi/thiếu, fingerprint, nhóm, ảnh trùng split và
preprocessing giữ nguyên dữ liệu nguồn. Kết quả smoke test không đại diện cho
accuracy trên URPC. Trạng thái chạy thật xem ở
[nhật ký triển khai](docs/research/pipeline-status.md).
