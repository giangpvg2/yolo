# Trạng thái triển khai pipeline

Ngày cập nhật: 2026-11-25.

**Trạng thái: đã sửa bộ dữ liệu do người dùng chỉ định ở `archive/URPC2020`.**
Chưa train full trên dữ liệu thực; chưa triển khai assignment hai view,
spectral diagnostic, wavelet hoặc M1–M4.

Cấu hình raw mặc định hiện trỏ trực tiếp vào nguồn này, giữ mapping
`holothurian, echinus, scallop, starfish` và split 5543/1200/800 ảnh.
Clip 3548 hộp vượt biên tối đa 1 pixel, loại 8 hộp diện tích 0; 2384 file
thay đổi có backup và nhật ký trước/sau. Audit sau sửa qua: 7543 ảnh, 57404
hộp, không thiếu nhãn. Có một nhóm ảnh trùng train/val, 17 cặp gần trùng
cần review và 160 ảnh nhãn rỗng cần kiểm tra mẫu. Chưa đạt D0 về leakage/nguồn.

Xem [urpc-data-check.md](urpc-data-check.md) và [PLAN.md](../../PLAN.md).
Artifact audit: `reports/urpc-source-audit-fixed/`; backup/repair log:
`reports/urpc-source-repair/`. Source YAML portable đã sửa tại chỗ;
metadata gốc không được xác nhận là nguồn/license hợp lệ.

## Đã hoàn thành

- Recipe train chung và CLI cho B0-Raw, B0-Enhanced, B1-Raw; scratch,
  640 px mặc định, seed cố định, có resume và environment/config snapshots.
- Dùng YOLOv8n-P2 từ Ultralytics 8.3.232, không cần YAML tùy chỉnh bị thiếu.
- COCO AP/AR theo size, FP/image, FN_small/rate, FROC CSV, nhóm proposed/reviewed
  và kiểm tra fingerprint khi nạp lại predictions. ID/class mapping giữ nguyên.
- Audit class/box sizes, SHA-256/dHash, nhóm chất lượng ảnh và clutter,
  sample review và báo cáo ảnh lỗi. Không sửa nhãn/split hoặc tự Go/No-Go.
- Preprocessing sao chép dữ liệu thay vì di chuyển, không ghi đè output,
  xử lý ảnh lồng thư mục; sinh data YAML portable.
- Khóa dependency và bổ sung hướng dẫn ở `README.md`, protocol ở `docs/pipeline.md`.

## Kiểm chứng

- 30 kiểm thử trên dữ liệu giả đã qua, gồm COCO perfect/empty/negative cases,
  matching, nhãn lỗi, làm tròn biên, dữ liệu thay đổi, ảnh trùng và preprocessing.
- YOLOv8n và YOLOv8n-P2 đều chạy thành công 1 epoch CPU ở 64 px trên dataset
  giả 2 ảnh train/2 ảnh val, lưu checkpoint và tự predict/evaluate.
- Audit CLI, grouped eval CLI và prediction replay chạy thành công;
  metric theo nhóm của checkpoint evaluation và replay khớp nhau.
- Môi trường kiểm chứng: Python 3.13.9, Ultralytics 8.3.232,
  Torch 2.9.1+cpu, Torchvision 0.24.1+cpu, pycocotools 2.0.11.
  Cặp Torch/Torchvision được cài riêng trong `.venv`; môi trường toàn cục
  trước đó không tương thích. Đây không phải kết quả train trên URPC thật.

## Lịch sử: export cũ đã kiểm tra (không dùng cho lượt mới)

Export `URPC.v1i.yolov8` được phát hiện trong lúc triển khai. Metadata của
export khai báo 5 lớp, khác cấu hình URPC2020 4 lớp ban đầu. Cấu hình riêng:
`configs/data_urpc_local.yaml`, giữ nguyên ID của nguồn.

| Split | Ảnh | Hộp | Fish | Scallop | Seacucumber | Seaurchin | Star-fish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 610 | 3588 | 0 | 610 | 692 | 1577 | 709 |
| val | 609 | 3277 | 1 | 264 | 651 | 1773 | 588 |

- Chưa có test trong export này; chưa xác minh đây là benchmark URPC2020 chuẩn.
- Audit: không thiếu file nhãn, 0 nhóm ảnh trùng hash, 25 cặp dHash gần trùng
  giữa split cần kiểm tra mắt. Hash không chứng minh độc lập nguồn cảnh quay.
- Có 251 hộp với sai số export ở biên, tối đa 0,005 pixel. Parser chấp nhận
  sai số không quá 0,01 pixel, giữ nguyên tọa độ và thống kê trong audit;
  không tự cắt hộp hay sửa nhãn. Test xác nhận sai lệch lớn hơn bị từ chối.
- Fish có 0 hộp train và 1 hộp val: cần kiểm tra nhãn và protocol trước khi
  diễn giải macro mAP; không tự đổi nhãn hoặc loại bỏ lớp.
- Artifact audit đầy đủ: `reports/urpc-local/` (generated, không commit).
  Chưa train full hoặc báo accuracy trên dữ liệu thực.

## Việc tiếp theo và cách tiếp tục

1. Đọc `PLAN.md`, báo cáo nguồn mới và audit trong `reports/urpc-source-audit-fixed/`.
2. Review ảnh trùng train/val và 17 cặp gần trùng; quyết định version/split
   phát triển, không sửa split nguồn âm thầm.
3. Xác minh metadata nguồn/license, xem mẫu ảnh nhãn rỗng và khóa nhóm ảnh.
4. Cài dependency/backend GPU đúng README; `.venv` cũ hiện không còn trong
   workspace. Lượt này dùng Python hiện có và pycocotools cài tạm riêng để test.
5. Chạy baseline chỉ sau khi chốt protocol; chưa tiếp tục module nghiên cứu.

Các file triển khai: `train.py`, `eval.py`, `dataset_audit.py`, `pipeline/`,
`configs/`, `tests/`, `requirements*.txt`, `preprocessing_data/` và tài liệu
liên quan. Artifact smoke test nằm trong `runs/pipeline-smoke/`, bị Git ignore;
không dùng các metric trên dữ liệu giả làm số liệu nghiên cứu.

## Nhật ký

- 2026-11-25: sửa nguồn `archive/URPC2020`, YAML/class mapping mặc định;
  clip 3548 hộp và bỏ 8 hộp diện tích 0 trong 2384 file đã backup. Audit qua
  7543 ảnh/57404 hộp, 39 tests pass; một nhóm trùng train/val và 17 cặp gần
  trùng cần review. Không train và không tự đổi split.
- 2026-11-25: chọn `archive/URPC2020` theo yêu cầu; sửa YAML và 2384 file
  annotation có backup, audit đầy đủ qua, thêm script repair tái lập và 9
  kiểm thử; toàn bộ 39 tests pass. Không train, không sửa split nguồn.


- 2026-11-25: người dùng chọn `urpc/`; kiểm tra 7543 ảnh/7543 nhãn TXT rỗng,
  không có metadata lớp trong bộ này. Lưu báo cáo, ignore dataset và chờ
  vị trí nhãn gốc; không sửa nhãn, không train và không thay PLAN đang có sửa tay.
- 2026-11-25: theo yêu cầu mới, viết `PLAN.md` từ xử lý dữ liệu đến baseline,
  chẩn đoán, chọn nhánh, prototype, đánh giá và báo cáo; giữ trạng thái code
  tạm dừng và không chạy thêm train.
- 2026-11-25: triển khai nền train/eval/audit, 30 tests pass, smoke test CPU
  B0/B1 và audit dữ liệu thực; dừng theo yêu cầu người dùng và lưu bàn giao.

Giữ nguyên `docs/research/assignment-roadmap.md` và các nghiên cứu hiện có.
Không tạo `.ai/` vì repo chưa có thư mục này và hướng dẫn dự án yêu cầu
tiếp tục dùng bối cảnh hiện hữu. Các tài liệu nghiên cứu, PDF và file `vs.md`
đã có thay đổi từ trước được giữ ngoài commit triển khai.
