# Pipeline baseline và kiểm toán dữ liệu

## Phạm vi

Triển khai phần nền dùng chung của lộ trình: B0-Raw, B0-Enhanced,
B1-Raw, COCO evaluation và dataset audit. Chưa triển khai wavelet,
assignment hai view hoặc M1–M4; chưa có kết quả xác nhận giả thuyết.

## Thiết kế

- `train.py`: đọc recipe dùng chung, chọn baseline, kiểm tra dữ liệu,
  train và tự đánh giá checkpoint tốt nhất trên validation.
- `pipeline/data.py`: đọc YOLO detection labels, kiểm tra class/box và
  tạo COCO ground truth với ID ổn định theo thứ tự đường dẫn trong split.
- `eval.py`, `pipeline/evaluation.py`: dự đoán trên ảnh gốc hoặc đọc
  COCO predictions đã lưu; xuất JSON/CSV cho toàn split và từng nhóm.
- `dataset_audit.py`: thống kê lớp/kích thước, hash ảnh, cảnh báo trùng
  split và đề xuất nhóm chất lượng ảnh/nền để người dùng kiểm tra.

Đường dẫn `path` trong data YAML được giải tương đối với file YAML.
Các đường dẫn split được giải tương đối với `path`. Pipeline dùng cấu trúc
YOLO `images/<split>` và `labels/<split>`, không phụ thuộc datasets_dir
trong cấu hình người dùng của Ultralytics. File nhãn rỗng là ảnh âm hợp lệ;
file nhãn thiếu là lỗi trừ khi bật tùy chọn cho phép rõ ràng.
Audit gom các ảnh lỗi vào `validation_issues.json` và dừng trước khi tính
thống kê/nhóm; train/eval từ chối dữ liệu không hợp lệ. Không tự cắt hộp hoặc
sửa nhãn nguồn.
Chấp nhận sai số export tại biên tối đa 0,01 pixel (theo từng kích thước ảnh),
giữ nguyên tọa độ và báo `border_rounding_boxes` trong audit.

## Protocol

- Mặc định 640 px, 100 epochs, batch 16, seed 0, deterministic và scratch.
  Không tải pretrained tự động. B0/B1 dùng model YAML được đóng gói cùng
  Ultralytics 8.3.232. Khóa recipe trước khi so sánh; có thể đổi seed qua CLI.
- `Enhanced` là Gray World + CLAHE theo script hiện có, không phải CLAHE riêng.
- COCO AP/AR dùng bbox area ở ảnh gốc, IoU 0.50:0.95, maxDets 1/10/100.
  FP/image và FN_small dùng matching một-một theo lớp tại confidence/IoU
  ghi trong kết quả; NMS IoU là tham số riêng. FP/image gồm mọi size.
- Không dùng confidence đếm lỗi cao để lọc predictions đầu vào COCO AP.
- Xuất `froc.csv` trên toàn split để so FP/image tại cùng recall trên development data.
- Severity/clutter tự động là proxy, không phải ground truth vật lý.
  Ngưỡng cố định trong `configs/audit.yaml`, không fit theo test.
  Chỉ dùng nhóm sau khi kiểm tra thủ công cho các tuyên bố khoa học.
- Test không tham gia chọn checkpoint, ngưỡng hay hard-negative mining.
  Kiểm chứng ảnh trùng/gần trùng trước khi gọi dataset thứ hai độc lập.

## Công việc và tiêu chí hoàn tất

1. Cấu hình raw/enhanced và train CLI (P0): chạy được B0/B1 và lưu recipe.
2. COCO và đếm lỗi (P0): kiểm thử perfect/empty/duplicate/wrong-class cases.
3. Audit (P0): xuất object CSV, image manifest, group JSON và báo cáo Markdown.
4. Kiểm tra tích hợp trên dữ liệu giả (P0): train CPU ngắn, predict, eval, audit.
5. Chạy thật trên URPC/DUO và GPU (chưa thực hiện): cần dữ liệu và tài nguyên.

Hash chính xác và dHash chỉ phát hiện một phần ảnh trùng; không chứng minh
độc lập nguồn cảnh quay. Audit không tự thay nhãn, chia lại split hoặc quyết
định Go/No-Go. Thời gian train thật cần đo trên GPU trước khi lập lịch.
