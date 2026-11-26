# Kiểm tra dataset urpc

## Nguồn hiện tại: archive/URPC2020

Người dùng cung cấp nguồn này sau khi phát hiện `urpc/` có nhãn rỗng và yêu
cầu sửa dữ liệu tại đây. Pipeline raw nay trỏ trực tiếp vào nguồn đã chọn,
không dùng export năm lớp hoặc thư mục `urpc/` cũ.

Class IDs giữ theo YAML nguồn: `0=holothurian`, `1=echinus`, `2=scallop`,
`3=starfish`. Không remap số lớp dựa trên cấu hình cũ.

| Split | Ảnh | Hộp sau sửa | holothurian | echinus | scallop | starfish | Nhãn rỗng |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5543 | 41437 | 5536 | 22342 | 6718 | 6841 | 88 |
| val (valid) | 1200 | 9388 | 461 | 4231 | 3196 | 1500 | 47 |
| test | 800 | 6579 | 373 | 2047 | 3236 | 923 | 25 |

Đã sửa:

- `archive/URPC2020/data.yaml`: đường dẫn portable, giữ mapping và layout.
  Bản gốc ở `reports/urpc-source-repair/data.original.yaml`; metadata Roboflow
  của bản gốc có tên project không khớp nội dung bốn lớp nên không dùng để
  khẳng định provenance/license. Cần xác minh nguồn benchmark độc lập.
- `configs/data_urpc_raw.yaml`: trỏ vào `archive/URPC2020`, split `train`,
  `valid`, `test`. `configs/data_urpc_clahe.yaml` giữ cùng ID lớp cho nhánh
  tăng cường sẽ chuẩn bị sau; chưa tạo dữ liệu enhanced ở lượt này.
- Chạy `preprocessing_data/repair_labels.py --apply`: 2384 file nhãn sửa,
  3548 hộp được clip về khung ảnh, 8 dòng có diện tích bằng 0 bị loại.
  Hộp vượt biên tối đa 1 pixel; không đổi class hoặc sửa hộp hợp lệ.
- Toàn bộ file nhãn thay đổi được sao lưu trước khi ghi; từng dòng trước/sau
  và SHA-256 có trong `reports/urpc-source-repair/labels/repair.json`.
  Backup ở `reports/urpc-source-repair/labels/original_labels/`.

Tổng sau sửa: 7543 ảnh, 57404 hộp. Không thiếu file nhãn; ảnh/split nguồn được
giữ nguyên. Nhãn rỗng nguồn được giữ nguyên, chưa kết luận mọi ảnh này là
background thật; cần xem mẫu để kiểm tra nhãn thiếu.

Audit sau sửa qua đủ ba split; 39 tests pass. Có 17 cặp gần trùng giữa
các split cần xem mắt; nhóm ảnh tự động chưa được review thủ công.

Kiểm tra hash tìm thấy hai nhóm ảnh trùng chính xác, trong đó có một nhóm
train/val: `train/images/000002.jpg` và `valid/images/000002.jpg`. Chưa xóa hay
chuyển ảnh này, không tự thay protocol nguồn. Audit có 17 cặp gần trùng cần review. Cần xử lý leakage và kiểm tra
gần trùng trước thí nghiệm chính; dữ liệu hợp lệ về box chưa đủ đạt D0.

Audit ban đầu thất bại được giữ tại `reports/urpc-source-audit/`. Audit sau
sửa đã qua trên toàn bộ ảnh/nhãn, ghi tại `reports/urpc-source-audit-fixed/`. Chưa train ở lượt sửa dữ liệu.

## Lịch sử: thư mục urpc có nhãn rỗng

Ngày kiểm tra: 2026-11-25. Người dùng chọn `urpc/` thay cho export Roboflow
trước đó. Chưa xác minh phiên bản benchmark, nguồn hoặc thứ tự lớp.

| Split | Ảnh | File nhãn TXT | Nhãn rỗng 0 byte |
|---|---:|---:|---:|
| train | 6034 | 6034 | 6034 |
| val | 754 | 754 | 754 |
| test | 755 | 755 | 755 |

Tổng: 7543 ảnh, 7543 file nhãn rỗng, không có hộp được gán nhãn. Mỗi ảnh
có file TXT cùng stem; không có nhãn mồ côi hoặc stem ảnh trùng giữa split.
Kiểm tra stem không thay cho kiểm tra nội dung ảnh trùng/gần trùng.

Không thấy YAML, JSON, XML, class names hay metadata trong thư mục dataset.
Các ảnh mẫu đọc được bằng OpenCV; chưa kiểm tra giải mã toàn bộ ảnh hoặc
audit hash đầy đủ. Không chạy train và không sửa dữ liệu nguồn.

## Việc cần làm trước khi dùng

1. Xác định vị trí nhãn gốc XML/JSON/TXT hoặc archive và thứ tự lớp.
2. Xác minh dữ liệu thuộc benchmark/phiên bản nào và nguồn chia train/val/test.
3. Chuyển annotation sang YOLO nếu cần, trong bản dữ liệu mới; lưu nhật ký.
4. Tạo data YAML đúng mapping, audit nhãn/hộp và ảnh trùng/gần trùng.
5. Chỉ chạy baseline sau khi có đối tượng trong train và protocol được chốt.

Không suy ra class IDs từ tên URPC hoặc dùng mapping năm lớp của export cũ.
Không coi toàn bộ nhãn rỗng là 7543 ảnh background hợp lệ. Kiểm tra chặn
training split không có đối tượng của `train.py` cần được giữ nguyên.

Vấn đề Fish và 25 cặp dHash của `URPC.v1i.yolov8` là kết quả dataset cũ,
không áp dụng cho `urpc/`. Các phần dataset trong PLAN sẽ cần cập nhật khi
có metadata của bộ mới; PLAN đang có chỉnh sửa của người dùng được giữ nguyên.
