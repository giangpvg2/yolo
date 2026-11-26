# Kế hoạch từ đầu: nghiên cứu phát hiện vật thể nhỏ dưới nước


**Mục tiêu:** xây một quy trình có thể tái lập để xác định lỗi của detector,
kiểm chứng một cơ chế cải thiện và tạo kết quả đủ căn cứ cho báo cáo/paper.
Chưa chốt nơi nộp, thời hạn hoặc ngân sách GPU. Không đặt trước kết luận
phương pháp sẽ mới, sẽ tăng AP hoặc phải có đủ một bộ module.

**Trạng thái hiện tại:** đã có nền train/eval/audit; đang chuẩn hóa bộ dữ liệu
`archive/URPC2020` theo chỉ định mới. Chưa train đầy đủ hoặc triển khai module
nghiên cứu. Các phần đã làm sẽ được tái sử dụng, không viết lại.

## 1. Câu hỏi nghiên cứu và phạm vi

Câu hỏi chính: vật thể nhỏ dưới nước bị bỏ sót hoặc báo nhầm do đâu, và can
thiệp tối thiểu nào cải thiện được detection trên ảnh thực ở chi phí chấp nhận được?

Hai giả thuyết cần phân biệt:

- **H-A — Assignment:** suy giảm ảnh làm positive assignments hoặc chất lượng
  supervision không ổn định; tham khảo một view ít suy giảm hơn có thể giúp.
- **H-F — Feature/tần số:** P2 giữ cả chi tiết vật thể lẫn nền rối; thông tin
  phổ và ngữ cảnh tầng sâu có thể giúp điều tiết chi tiết có ích.

Ưu tiên khảo sát H-A theo hướng đã ghi trong
[assignment-roadmap](docs/research/assignment-roadmap.md). H-F là nhánh thay
thế có điều kiện, chưa phải hướng đã chọn. Chỉ phát triển **một nhánh chính**
sau chẩn đoán. Không ghép assignment, wavelet, enhancement, teacher và nhiều
attention block vào cùng prototype đầu tiên.

`assignment-roadmap.md` giữ thiết kế chuyên sâu của nhánh A; mô tả code ban
đầu trong file đó là lịch sử trước triển khai. Dùng PLAN và pipeline-status
để xác định trạng thái repo hiện tại.

Phạm vi đầu tiên: YOLOv8n và YOLOv8n-P2, ảnh raw theo nguồn dataset,
input 640, nhãn detection thật. Không coi raw là ảnh sạch về mặt vật lý.
Kiến trúc khác và dataset thứ hai được xét sau khi prototype có tín hiệu.

## 2. Điểm xuất phát thực tế

### 2.1. Những gì đã có

| Thành phần | Trạng thái | Vị trí |
|---|---|---|
| Train B0-Raw, B0-Enhanced, B1-Raw với cùng recipe | Đã triển khai | `train.py`, `configs/train.yaml` |
| Cấu hình P2 hợp lệ từ package đã pin | Đã chạy thử CPU | `yolov8n-p2.yaml` trong Ultralytics |
| COCO AP/AR theo size, FP/image, FN_small, FROC | Đã triển khai và kiểm thử | `eval.py`, `pipeline/evaluation.py` |
| Audit nhãn, kích thước hộp, hash, nhóm ảnh đề xuất | Đã chạy trên dữ liệu thực | `dataset_audit.py` |
| Preprocessing và data YAML portable | Đã triển khai | `preprocessing_data/`, `configs/` |
| Kiểm thử tự động | 39 tests pass sau bổ sung sửa nhãn | `tests/` |
| Smoke test train → checkpoint → predict → eval | B0/B1, 1 epoch CPU trên dữ liệu giả | `runs/pipeline-smoke/` |
| Train đầy đủ, GPU, resume khi Colab ngắt | Chưa kiểm chứng | Công việc tiếp theo |
| Assignment hai view, spectral diagnostic, wavelet | Chưa triển khai | Chỉ làm sau cổng quyết định |

Nền code được lưu ở commit `f4da791`. Chi tiết bàn giao ở
[pipeline-status](docs/research/pipeline-status.md); hướng dẫn chạy hiện có ở
[README](README.md), protocol ở [pipeline](docs/pipeline.md).

### 2.2. Dữ liệu đang có

Nguồn hiện được chọn: `archive/URPC2020`, cấu hình `configs/data_urpc_raw.yaml`.
Giữ class IDs từ YAML nguồn: `0=holothurian, 1=echinus, 2=scallop, 3=starfish`.

| Split | Ảnh | Hộp sau sửa | holothurian | echinus | scallop | starfish |
|---|---:|---:|---:|---:|---:|---:|
| train | 5543 | 41437 | 5536 | 22342 | 6718 | 6841 |
| val (valid) | 1200 | 9388 | 461 | 4231 | 3196 | 1500 |
| test | 800 | 6579 | 373 | 2047 | 3236 | 923 |

Đã sửa YAML portable, clip 3548 hộp vượt biên và loại 8 hộp diện tích 0;
2384 file nhãn thay đổi đều có backup và nhật ký SHA-256 trước/sau ở
`reports/urpc-source-repair/labels/`. Giữ nguyên ảnh, split và class ID.

Có hai nhóm ảnh trùng hash, trong đó một nhóm giữa train/val. Cần xử lý
leakage và review gần trùng trước D0; chưa coi dữ liệu đã sẵn sàng cho paper.
Có 160 file nhãn rỗng nguồn được giữ nguyên, cần kiểm tra nhãn thiếu qua mẫu.
Metadata Roboflow trong YAML gốc chưa được xác minh, không dùng để kết luận
nguồn hoặc license. Xem [kiểm tra dữ liệu](docs/research/urpc-data-check.md).

`URPC.v1i.yolov8` năm lớp và `urpc/` nhãn rỗng là các bản đã kiểm tra trước;
vấn đề Fish và 25 cặp dHash của export cũ không áp dụng cho nguồn hiện tại.
Artifact audit mới: `reports/urpc-source-audit-fixed/`. Giữ bản sao ngoài
phiên Colab. Không dùng smoke metrics trên dữ liệu giả làm kết quả nghiên cứu.

## 3. Quy tắc thực hiện

- Raw là nhánh chính. `B0-Enhanced` hiện là **Gray World + CLAHE**, không phải
  CLAHE riêng. Mỗi biến thể ảnh phải giữ cùng ảnh nguồn, nhãn và split.
- Train, validation và test có vai trò riêng. Không chọn module, ngưỡng hoặc
  checkpoint trên test; hard-negative mining chỉ dùng train.
- Mỗi run có ID, code/config/data fingerprint, seed, thời gian train,
  peak VRAM, checkpoint và metrics. Dùng cấu hình/version đã khóa.
- So sánh cùng resolution, khởi tạo, augmentation và lịch train. Với hai view,
  phải báo thêm số view/update và chi phí train, không chỉ số epochs.
- AP/AR chuẩn COCO dùng bbox ở ảnh gốc. Nhóm tiny theo diện tích tương đối
  hoặc pixel sau resize phải có tên và định nghĩa riêng.
- Protocol hiện tại dùng small area tối đa 1024 pixel², COCO IoU 0.50–0.95,
  maxDets 1/10/100. FN_small ở một ngưỡng không đồng nhất với AR_small trung
  bình nhiều IoU. Mặc định đếm lỗi tại confidence 0.25, match IoU 0.5;
  NMS IoU 0.7, prediction score floor 0.001 và prediction cap 300.
- FP/image và FN_small luôn kèm confidence, IoU matching và NMS settings.
  So FP ở cùng recall trên development data; khóa cách chọn ngưỡng trước test.
- Severity/clutter tự động chỉ là proxy. Giữ phân biệt `proposed` và `reviewed`.
- Đối chứng âm và kết quả không cải thiện vẫn được lưu. Mọi Go/No-Go do người
  thực hiện quyết định từ báo cáo, không do tên module hoặc script tự kết luận.

## 4. Giai đoạn 0 — Chốt bài toán và tiêu chuẩn thành công

**Công việc**

- Viết problem statement một trang: đối tượng sử dụng, lỗi cần giảm, giới hạn
  compute và điều kiện ảnh cần cải thiện.
- Chọn metric chính trước prototype: AP_small; dùng AP50–95 và FP/image tại
  recall tương đương để kiểm soát đánh đổi. AR_small và kết quả từng lớp là
  bằng chứng bổ sung bắt buộc nếu claim liên quan giảm bỏ sót.
- Đọc đối chứng theo nhánh chọn trong `docs/research/literature-review.md`
  và `research-options.md`. H-A cần đối chiếu TAL/coverage/consistency;
  H-F cần đối chiếu SET, DBS-Net, wavelet và semantic-gated fusion.
- Tách rõ: đã đọc full text/code, chỉ có abstract, chưa xác minh. Chưa tuyên bố novelty.

**Đầu ra dự kiến:** `reports/problem_statement.md`, bảng đối chiếu cơ chế.

**Hoàn thành khi:** xác định được một lỗi đo được, metric chính, đối chứng
gần nhất và câu hỏi mà pilot có thể bác bỏ. Các đường dẫn đầu ra chưa có
trong mục này và các mục sau là việc dự kiến, không phải file đã được tạo.

## 5. Giai đoạn 1 — Chốt dataset, class mapping và split

**Thứ tự công việc**

1. Dùng nguồn bốn lớp đã chọn, xác minh mapping với mẫu ảnh/nhãn. Kiểm tra
   160 ảnh có nhãn rỗng và nhật ký sửa box; không dùng lại mapping export cũ.
2. Xử lý ảnh trùng train/val và xem các cặp gần trùng của audit nguồn mới;
   gom các ảnh cùng scene/video khi có bằng chứng.
   Không tự xóa mọi cặp theo dHash. Ghi quyết định giữ/nhóm/loại và lý do.
3. Xác minh nguồn và phiên bản dataset. Ưu tiên benchmark có split chính thức
   đã kiểm chứng cho thí nghiệm chính; metadata Roboflow trong YAML gốc
   chưa xác minh nên không được dùng để khẳng định nguồn/split chính thức.
4. Nếu phải tự chia, chia theo nhóm nguồn/scene trước khi augmentation, khóa
   seed và manifest. Tỷ lệ 70/15/15 chỉ là phương án tham khảo; số lượng thực
   phụ thuộc scene và độ phủ lớp. Không âm thầm thay split chính thức.
5. Giữ một test chưa dùng cho quyết định mô hình. Không đổi tên một validation
   đã dùng để tuning thành test rồi coi như chưa quan sát.
6. Audit lại bản đã chốt; lưu hash, class coverage và thống kê size từng split.
   Review một mẫu phân tầng severity/clutter trên development set, khóa định
   nghĩa/threshold trước khi gán nhóm cho test.

**Đầu ra dự kiến:** dataset copy có version, `configs/data_dev.yaml`,
`reports/data_decisions.md`, manifest và báo cáo audit phiên bản mới.

**Cổng D0 — Được chạy thí nghiệm chính khi:** nguồn/class mapping rõ, hộp và
nhãn rỗng đã được kiểm tra, trùng/gần trùng được xử lý, split và test được khóa.
Nếu chưa đạt, chỉ kiểm tra kỹ thuật; không báo accuracy như kết quả benchmark.

## 6. Giai đoạn 2 — Kiểm tra nền code và môi trường GPU

- Review code đã có: resolution/path/ID mapping, class names checkpoint,
  bbox và area gốc, matching một-một, tập không có GT/prediction, undefined metrics.
- Bổ sung export metric từng lớp; hiện evaluator chủ yếu xuất metric tổng,
  theo size và theo nhóm ảnh. Kiểm tra cap COCO 100 có ảnh hưởng cảnh đông
  không; nếu thêm AP@300 phải đặt tên riêng, giữ số COCO chuẩn để đối chiếu.
- Cài backend GPU khớp Torch/Torchvision đã pin. `.venv` hiện tại là CPU;
  không sao chép nguyên môi trường này để train trên A100/Colab.
- Chạy 1–3 epoch GPU trên bản dữ liệu hợp lệ, đo thời gian/epoch, peak VRAM,
  NaN/Inf và khả năng lưu/tải checkpoint.
- Thử ngắt rồi resume checkpoint còn optimizer state. Kiểm tra epoch/update
  tiếp tục đúng; không yêu cầu checkpoint đã hoàn tất và bị strip optimizer
  phải resume được. Ghi hạn chế tái lập RNG nếu có.
- Lưu checkpoint/artifact ở nơi tồn tại sau khi phiên Colab kết thúc.

**Recipe xuất phát:** 640 px, 100 epochs, batch 16, seed 0, SGD,
scratch và AMP=false theo `configs/train.yaml`. Đây chưa phải lịch đã chứng
minh hội tụ. Dùng pilot để chốt batch/lịch/AMP; thay thì áp dụng lại đồng nhất
cho mọi baseline, không đổi riêng phương án mới.

**Cổng D1:** B0/B1 train, resume và eval GPU chạy đúng; môi trường và recipe
được lưu. Smoke test CPU đã qua không thay cho cổng này.

## 7. Giai đoạn 3 — Chạy baseline và lập ngân sách

| ID | Cấu hình | Mục đích |
|---|---|---|
| B0-Raw | YOLOv8n, raw | Baseline gốc |
| B1-Raw | YOLOv8n-P2, raw | Đo tác động thực của P2 |
| B0-Enhanced | YOLOv8n, Gray World + CLAHE | Đối chứng preprocessing, ưu tiên sau B0/B1 |

Chạy B0/B1 seed 0 trước. Kiểm tra learning curves và nhóm lỗi. Nếu kết luận
nhạy với một seed, lặp seed 1, 2 cho các baseline liên quan trước khi quyết
định hướng. Screening rút ngắn chỉ dùng chọn cấu hình, không trộn bảng với full training.

Ghi AP50–95, AP_small, AR_small, từng lớp, FP/image, FN_small/rate,
severity/clutter đã review, thời gian và VRAM. Tất cả metric phải truy về
cùng phiên bản dataset và checkpoint.

Ước lượng GPU-hours = tổng thời gian các run đo từ pilot + phần diagnostic,
evaluation và lượt chạy lại. Không cam kết thời gian train chỉ từ tên GPU.

**Cổng D2:** baseline hội tụ đủ cho câu hỏi nghiên cứu, số liệu/nhãn đáng tin,
ngân sách đủ cho đối chứng. P2 không mặc định tốt hơn; nếu không có lợi ích,
kiểm tra nguyên nhân rồi cân nhắc nghiên cứu trên B0.

Pilot bắt đầu seed 0. Nếu chênh lệch nhỏ so với mức có ý nghĩa đã định hoặc
thứ hạng thay đổi theo checkpoint/subset development, chạy thêm seed 1/2
cho B0/B1 trước D3. Một seed không đủ kết luận về độ ổn định qua seed.

## 8. Giai đoạn 4 — Chẩn đoán và chọn một hướng

### 8.1. H-A: assignment dưới suy giảm ảnh — ưu tiên khảo sát

- Đọc đúng implementation assigner/loss của phiên bản đã pin. Thêm logger
  dense trước NMS, định danh điểm bằng `(level, row, column)` và GT tương ứng.
- Tạo A/B có cùng hình học và GT; A là ảnh gốc, B có blur/contrast/color cast
  nhẹ–vừa. Tách từng loại trước, không coi đây là mô phỏng vật lý chính xác.
- Dùng checkpoint cố định. Kiểm tra logger tái hiện đường assignment gốc;
  ghi rõ trạng thái train/eval/BatchNorm. A=B phải cho kết quả nhất quán.
- Lưu positive count, tỷ lệ GT không có positive, overlap/churn assignment,
  chất lượng positive và cạnh tranh GT. Hai tập cùng rỗng phải được tính là
  thiếu coverage, không coi là ổn định tốt.
- Phân tầng/kiểm soát size, lớp, confidence, số positive và candidate hình học.
  Phân biệt lỗi trước/sau NMS. FN trên ảnh mờ chưa đủ kết luận lỗi assignment.

**Đi tiếp H-A khi:** bất ổn/chất lượng assignment có tín hiệu lặp lại liên quan
đến lỗi thật ngoài các yếu tố trên, và có can thiệp nhỏ kiểm chứng được.
Nếu chủ yếu thiếu candidate hình học, phải dùng coverage-only làm đối chứng.

### 8.2. H-F: P2, clutter và tín hiệu phổ — nhánh có điều kiện

- Xác định TP_small, FP mới của B1 so với B0 bằng matching theo ảnh/lớp;
  kiểm tra một mẫu khoảng 50 FP mới về nhãn thiếu, không coi mẫu này là ước
  lượng chính xác tỷ lệ nhãn thiếu toàn dataset.
- Trích P2, Haar LL/LH/HL/HH và P3 phù hợp. So khả năng phân biệt TP/FP của
  confidence+size, P3, và P3+spectral trên ảnh/scene giữ riêng để kiểm chứng.
- Kiểm soát size/lớp/confidence; thử can thiệp dải tần đơn giản, có phép
  nghịch đảo và identity control để kiểm tra ảnh hưởng tới detection.
- Không mặc định HH là noise hay LL của feature là màn mù vật lý. Cùng
  stride không tự bảo đảm alignment; tránh vòng phụ thuộc P2↔P3.

**Đi tiếp H-F khi:** có lỗi P2 phù hợp giả thuyết, spectral bổ sung thông tin
ngoài các đối chứng và can thiệp cho thấy đánh đổi AP/FP có thể cải thiện.
Phân bố phổ khác nhau riêng lẻ chưa đủ để xây module.

### 8.3. Quyết định D3

| Kết quả | Hành động |
|---|---|
| H-A có bằng chứng, chưa có đối chứng đơn giản giải thích hết | Prototype assignment |
| H-A yếu, H-F đạt điều kiện | Chuyển sang prototype tần số |
| Cả hai có bằng chứng | Chọn một theo độ mới, chi phí và khả năng kiểm chứng |
| Cả hai yếu hoặc nhãn/split chưa chắc | Sửa dữ liệu/protocol hoặc dừng hướng module |

**Đầu ra dự kiến:** `reports/diagnostics/`, hình lỗi và một decision note ghi
bằng chứng thuận/nghịch, nhánh chọn, đối chứng và ngân sách validation.
Không bắt buộc triển khai cả hai bộ diagnostic đầy đủ trước khi chọn nhánh;
ưu tiên H-A, mở H-F khi cần và trong ngân sách đã định.

## 9. Giai đoạn 5 — Prototype có đối chứng

### 9.1. Nếu chọn H-A

Ma trận dưới đây là phương án khi D2 giữ P2. Nếu D2 chọn B0, giữ E0/E1 làm
đối chiếu kiến trúc, nhưng đổi nền của E2–E7 đồng nhất sang B0 trước khi chạy.
Cập nhật ma trận và ngân sách trong decision note, không so prototype B0 với
đối chứng hai view P2 như thể chỉ khác assignment. Nếu đổi nền, cũng xem lại
phạm vi H-F thay vì giữ nguyên claim về nhánh detection P2.

| ID | Biến thể | Tách tác động nào? |
|---|---|---|
| E0 | B0-Raw | Kiến trúc gốc |
| E1 | B1-Raw | P2 |
| E2 | P2 + suy giảm một view | Augmentation |
| E3 | P2 + A/B, TAL độc lập, trung bình hai loss | Hai view và thêm compute |
| E4 | E3 + prediction consistency | Consistency thông thường |
| E5 | E3 + chuyển assignment A sang B trực tiếp | Chép lựa chọn đơn giản |
| E6 | E3 + chuyển có điều kiện, fallback TAL | Cơ chế đề xuất |
| E7 | E3 + coverage-only | Tăng/phân bố positive |

Triển khai E3 trước để làm đường code chung. E5/E6 luôn dùng GT thật của B
làm target, không thay bằng predicted box của A. Detach tín hiệu chọn mẫu;
giữ normalization loss và policy candidate/top-k có đối chứng. Không ép
giữ assignment khi B đã mất bằng chứng. Chưa thêm EMA teacher ở prototype đầu.

Kiểm tra A=B, tắt can thiệp, empty labels, GT chồng lấp, điểm cạnh tranh,
không có candidate, loss/gradient hữu hạn, resume và graph inference.

### 9.2. Nếu chọn H-F

Thứ tự: B1 → Haar + fusion đơn giản → scalar gate → gate theo band → thêm
P3 conditioning nếu có bằng chứng. So với Conv/attention có ngân sách tương
đương; kiểm tra đường trả feature về P2. Local degradation conditioning và
train-only supervision chỉ thêm từng phần khi phiên bản trước đã có ích.

GT boxes là giám sát thô; vùng ngoài GT chưa chắc là nền thật. Hard negatives
phải lấy từ train và kiểm soát nhãn thiếu. Không mặc định phải xây đủ M1–M4.

**Cổng D4:** prototype cải thiện detection thực so với đối chứng mạnh, không
chỉ giảm loss hay làm hình gate đẹp. Khóa mức cải thiện tối thiểu có ý nghĩa
và mức suy giảm metric phụ chấp nhận được trước khi tuning prototype, dựa
trên biến thiên baseline và yêu cầu compute. Nếu gain chỉ do thêm view,
coverage hoặc tham số, sửa claim hoặc dừng cơ chế đề xuất.

## 10. Giai đoạn 6 — Đánh giá đầy đủ

- Chạy seeds 0, 1, 2 cho các cấu hình chính; báo mean ± std và chênh lệch
  cùng seed. Với H-A, nhóm dự kiến E0/E1/E3/E5/E6/đối chứng mạnh nhất là
  **18 full runs** trên dataset chính, chưa tính screening và bộ thứ hai.
  Điều chỉnh ma trận trước khi chạy, không cộng trùng nếu đối chứng trùng ID.
- Ước lượng uncertainty bằng bootstrap theo ảnh/scene phù hợp; tính lại AP
  trên mẫu bootstrap, không lấy trung bình AP của từng ảnh hay từng hộp.
- Báo trade-off AP_small/AR_small/FP và từng lớp; subgroup không có GT ghi
  undefined, kèm số ảnh/GT để tránh diễn giải nhóm quá ít mẫu.
- Kiểm tra robustness trên mức/loại corruption giữ riêng khỏi tuning, đồng
  thời giữ kết quả ảnh thực là chính. Không để synthetic corruption thay cho
  bằng chứng detection dưới nước thực tế.
- Train riêng baseline/full model trên dataset thứ hai khi ngân sách cho phép.
  Kiểm tra trùng/nguồn giữa dataset, class spaces và nhãn thiếu. RUOD/DUO là
  ứng viên cần audit, không được mặc định độc lập hoàn toàn với URPC.
- Nếu đánh giá chuyển miền trực tiếp, xác định ánh xạ lớp và giao thức riêng;
  không gọi train lại trên dataset thứ hai là zero-shot generalization.
- Sau khi khóa phương pháp, kiểm tra trên một detector khác phù hợp để đo
  tính tổng quát. Đây là mở rộng theo ngân sách; không là lý do trì hoãn pilot.
- Đo params/FLOPs/latency trên cùng GPU, input, batch=1, precision và runtime;
  có warm-up và đồng bộ GPU. Tách inference thuần với end-to-end pre/postprocess.
  Nếu claim triển khai nhúng, thêm export và benchmark đúng thiết bị đích.

**Cổng D5:** cải thiện lặp lại, đối chứng/compute công bằng, không có lỗi split,
claim phù hợp phạm vi đã kiểm chứng. Graph không thêm nhánh train phụ không
đồng nghĩa FPS bằng nhau tuyệt đối hoặc toàn hệ thống có chi phí bằng 0.

## 11. Giai đoạn 7 — Viết báo cáo/paper và đóng gói

1. Chốt bảng thí nghiệm, ablation, kết quả âm và failure cases trước phần claim.
2. Viết Experiments → Method → Introduction/Related Work → Abstract.
3. Chuẩn bị hình motivation từ diagnostic, kiến trúc train/inference và ví dụ
   định tính cùng ảnh/threshold. Không chọn hình test để điều chỉnh phương pháp.
4. Mỗi con số phải truy được về run ID, config, checkpoint và evaluator.
5. Đóng gói hướng dẫn tái lập, môi trường, split/hash, code và giới hạn dữ liệu.
   Nêu nguồn/license, nhãn thiếu, sample size, biến thiên seed và chi phí train.
6. Chỉ kết luận novelty/hiệu quả theo bằng chứng và đối chiếu đã đọc đầy đủ.

**Hoàn thành khi:** người khác chạy lại được protocol và mọi claim có kết quả
hỗ trợ. Kết quả pilot âm vẫn được ghi lại; không bắt buộc có paper nếu cơ chế
không vượt đối chứng.

## 12. Thứ tự công việc và thời gian tham khảo

Ước lượng cho một người, từ trạng thái hiện tại; thời gian chuẩn hóa code đã
làm không tính lại. Đây là ngày công, không phải cam kết thời gian GPU.

| Giai đoạn | Ưu tiên | Ước lượng | Phụ thuộc | Mốc hoàn tất |
|---|---|---|---|---|
| 0. Bài toán, đối chứng, tiêu chí | P0 | 0,5–1 ngày | — | Problem statement |
| 1. Nguồn dữ liệu, leakage, gần trùng, split | P0 | 2–4 ngày | Có dữ liệu phù hợp | D0 |
| 2. Review nền, metric từng lớp, GPU/resume | P0 | 1–2 ngày | D0 | D1 |
| 3. Baseline và ngân sách | P0 | 2–4 ngày + GPU | D1 | D2 |
| 4. Diagnostic và chọn nhánh | P0 | 3–5 ngày + GPU nếu cần | D2 | D3 |
| 5. Một prototype và đối chứng | P1 | 5–8 ngày + GPU | D3 | D4 |
| 6. Full evaluation | P1 | 7–12 ngày + GPU | D4 | D5 |
| 7. Báo cáo và gói tái lập | P1 | 3–5 ngày | D5 | Bản thảo/gói chạy lại |

Tổng khoảng **24–41 ngày công**, tương đương **5–9 tuần làm việc**, chưa tính
chờ GPU, bổ sung dữ liệu hoặc xử lý lại thí nghiệm thất bại. Các task ở bảng
là phân chia giai đoạn, không cộng thêm lần nữa. Khóa lại lịch sau pilot GPU;
nếu thiếu nguồn lực, giảm số mở rộng, giữ các đối chứng quyết định claim.

## 13. Checklist bắt đầu lại

- [x] Có train/eval/audit dùng chung và dependency đã pin.
- [x] Có 39 tests pass và smoke test CPU B0/B1 từ giai đoạn trước.
- [x] Có audit ban đầu trên export 1.219 ảnh.
- [x] Sửa/audit bộ nguồn mới: 7543 ảnh, 57404 hộp; lưu backup và nhật ký.
- [ ] Đọc trạng thái bàn giao và review code; chưa chạy train dài ngay.
- [x] Chọn nguồn `archive/URPC2020`, sửa YAML và box có backup/nhật ký.
- [ ] Kiểm tra trùng/gần trùng, nhãn rỗng và nguồn dataset hiện tại.
- [ ] Chốt class mapping, split/test và tạo config development có version.
- [ ] Chốt metric/đối chứng và kiểm tra GPU/resume.
- [ ] Chạy B0/B1, chẩn đoán, ghi quyết định D3.
- [ ] Chỉ sau đó triển khai một prototype.

### Lệnh hiện có để kiểm tra cấu hình

Chạy từ root sau khi kích hoạt môi trường đúng backend. Lệnh này không train
và không chứng nhận dataset đã sẵn sàng làm benchmark:

```bash
python train.py --baseline B1-Raw --data configs/data_urpc_raw.yaml --dry-run
```

### Lệnh baseline sau khi đã tạo config development và đạt D0/D1

`configs/data_dev.yaml` dưới đây **chưa có**; phải tạo từ quyết định dữ liệu
ở giai đoạn 1. Các lệnh sử dụng CLI đã triển khai, không phải lệnh đã chạy:

```bash
python train.py --baseline B0-Raw --data configs/data_dev.yaml --seed 0 --name dev-B0-s0 --device 0
python train.py --baseline B1-Raw --data configs/data_dev.yaml --seed 0 --name dev-B1-s0 --device 0
```

Mốc hành động gần nhất là **D0: chốt dữ liệu**, không phải xây module mới.
