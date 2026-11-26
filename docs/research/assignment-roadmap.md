# Lộ trình nghiên cứu assignment ổn định dưới suy giảm ảnh

Ngày lập: 2026-11-25. Trạng thái: kế hoạch khảo sát; chưa triển khai phương pháp, chưa có kết quả thực nghiệm xác nhận giả thuyết.

## 1. Phạm vi và mục tiêu

Nghiên cứu cách chọn positive anchor points khi huấn luyện YOLOv8-P2 bằng hai phiên bản của cùng ảnh. A là ảnh gốc hoặc ít suy giảm hơn; B được làm giảm chất lượng vừa phải, giữ nguyên tọa độ vật thể. Dùng bằng chứng từ A và ground truth thật để hỗ trợ assignment trên B. Không mặc định mọi điểm được chọn trên A phải được giữ trên B.

Thiết bị người dùng có: A100 qua Colab. Chưa chốt thời hạn, nơi nộp bài hoặc ngân sách GPU. Giả định một người triển khai; thời gian dưới đây là ước lượng công việc, cần điều chỉnh sau khi đo thời gian train và mức truy cập Colab thực tế.

Mục tiêu nghiên cứu: xác định assignment thay đổi dưới suy giảm có liên quan đến lỗi detection không; kiểm tra can thiệp vào assignment có tốt hơn augmentation, prediction consistency và tăng positive coverage không. Tính mới là giả thuyết cần đối chiếu full text, không phải kết luận đã xác nhận.

Phạm vi đầu tiên: YOLOv8n-P2, URPC2020, suy giảm photometric và blur nhẹ–vừa, ground truth có sẵn. Hoãn wavelet, physical teacher, EMA teacher, thay backbone và biến đổi hình học cho đến khi prototype tối thiểu có bằng chứng.

## 2. Trạng thái repo đã kiểm tra

- `train.py` tham chiếu `yolov8n_p2.yaml`, train ở 960 px, 100 epochs, batch 16 trên `URPC2020_CLAHE/data.yaml`.
- Chưa thấy file `yolov8n_p2.yaml` trong workspace đã kiểm tra; cần bổ sung hoặc sửa tham chiếu tới cấu hình hợp lệ.
- `requirements.txt` chưa khóa phiên bản Ultralytics và các phụ thuộc.
- `preprocessing_data/scripts.sh` giả định dữ liệu đã có train/valid/test và dùng thao tác di chuyển file. Cần kiểm tra dữ liệu thực tế, quyền sử dụng, nhãn test và bản sao trước khi chạy.
- `preprocessing_data/create_yaml.py` hiện tạo cấu hình cho dữ liệu CLAHE. Chưa có pipeline đánh giá assignment hoặc cấu hình train hai view trong repo đã kiểm tra.
- Không có `.ai/` tại thời điểm kiểm tra; dùng các tài liệu nghiên cứu hiện có và tài liệu này để lưu bối cảnh.
- Có thay đổi và file chưa được commit từ trước. Tài liệu này không thay đổi hoặc xác nhận các kết quả trong những file đó.

## 3. Những giả thuyết phải kiểm chứng

H1: suy giảm vừa phải làm chất lượng hoặc tập positive thay đổi nhiều hơn ở vật thể nhỏ so với nhóm lớn hơn, khi giữ cùng checkpoint và hình học.

H2: mức bất ổn/chất lượng positive có thông tin liên quan đến lỗi bỏ sót ngoài kích thước và confidence. Tương quan không chứng minh nguyên nhân; cần can thiệp có đối chứng ở giai đoạn tiếp theo.

H3: assignment có tham khảo A cải thiện AP/recall trên ảnh thực so với huấn luyện hai view độc lập, không chỉ giảm số lần đổi assignment.

H4: chuyển lựa chọn có điều kiện hiệu quả hơn chép assignment từ A sang B và hơn tăng positive coverage đơn thuần.

Nếu H1/H2 không có tín hiệu, không mặc định phát triển đủ pipeline; ghi lại kết quả và xem xét cơ chế lỗi khác. Nếu H3/H4 thất bại, thu hẹp claim hoặc dừng phương pháp.

## 4. Các giai đoạn và đầu ra

| Giai đoạn | Công việc | Đầu ra cần có | Công sức dự kiến |
|---|---|---|---|
| 0. Khóa phạm vi, đọc đối chứng | Đọc mã assigner/loss của phiên bản chọn; đối chiếu TAL, STAL, Consistent-Teacher, DCP-UDet | Problem statement, bảng cơ chế và điểm khác biệt còn cần xác minh | 2–3 ngày |
| 1. Chuẩn hóa dữ liệu và môi trường | Khóa môi trường, sửa cấu hình P2, kiểm tra nhãn/split và ảnh gần trùng | Manifest split, dataset audit, môi trường tái lập | 2–4 ngày |
| 2. Baseline và công cụ quan sát | Train YOLOv8n, YOLOv8n-P2; lưu checkpoint; ghi assignment | Baseline metrics, assignment logger và ảnh trực quan | 3–5 ngày |
| 3. Kiểm chứng hiện tượng | Đo A/B bằng checkpoint cố định; phân tầng size/lớp/suy giảm | Diagnostic report và quyết định tiếp tục/đổi giả thuyết | 3–5 ngày |
| 4. Prototype có đối chứng | Hai view độc lập, chép A sang B, chuyển có điều kiện | Trainer/assigner tối thiểu và kết quả screening | 4–7 ngày |
| 5. Đánh giá đầy đủ | Ablation, nhiều seed, bộ dữ liệu thứ hai, ảnh thực và robustness | Bảng kết quả, uncertainty, failure cases, chi phí | 7–12 ngày |
| 6. Viết và đóng gói | Viết phương pháp, kết quả, giới hạn; tái lập và export | Bản thảo, cấu hình, checklist tái lập | 4–6 ngày |

Tổng khung tham khảo khoảng 5–9 tuần làm việc tập trung; thời gian chờ GPU, chạy lại và phản biện có thể kéo dài thêm. Viết nhật ký, related work và phần protocol ngay từ đầu; phần kết luận chỉ viết theo kết quả đã có.

## 5. Protocol dữ liệu và baseline

- Dùng ảnh gốc làm nhánh nghiên cứu chính; CLAHE là thí nghiệm riêng sau đó. Không gọi ảnh gốc là ảnh sạch hoặc ground truth phục hồi.
- Xác định chính xác bản URPC2020 đang sử dụng và class mapping. Lưu đường dẫn tương đối của ảnh, split, nhãn và hash; kiểm tra ảnh hỏng, hộp sai, nhãn thiếu qua một mẫu thủ công.
- Chia dữ liệu trước khi sinh biến thể. Hai view của cùng ảnh luôn cùng split. Kiểm tra ảnh trùng/gần trùng, ưu tiên nhóm cùng video/cảnh nếu có metadata.
- Chỉ dùng train để tối ưu trọng số; dùng validation để chọn checkpoint, mức suy giảm và siêu tham số. Đóng băng protocol trước khi đánh giá test; không chọn ví dụ/threshold dựa vào test.
- Khóa Ultralytics, Python, PyTorch/CUDA và commit adapter. Đọc đúng mã ở phiên bản đã khóa, không sao chép API trên nhánh main rồi mặc định tương thích YOLOv8.
- So baseline ở cùng input, split, initialization/pretraining, lịch train và augmentation. Kiểm tra hiệu quả P2 trước khi coi nó là điều kiện bắt buộc.
- Pilot dùng YOLOv8n, một seed, cấu hình vừa VRAM. Đo thời gian epoch, VRAM và throughput rồi chọn input/batch. Cấu hình full phải chạy lại đồng nhất, không so AP giữa các resolution như thể chỉ khác phương pháp.
- Lưu config, seed, checkpoint, metrics và optimizer/scheduler/scaler/RNG state để resume khi Colab ngắt; kiểm tra khả năng khôi phục. Ghi hạn chế tái lập nếu môi trường không cho kết quả bitwise giống nhau.

## 6. Thiết kế phép đo A/B

A và B có cùng crop/resize/letterbox và cùng ground truth. Tạo B sau các biến đổi hình học chung; bắt đầu không dùng Mosaic/MixUp trong chẩn đoán. Trong train đối chứng, mọi phương pháp dùng cùng thiết lập augmentation và cùng thứ tự tạo cặp.

Tạo B bằng từng loại suy giảm riêng trước: contrast, color cast, blur; sau đó mới thử kết hợp. Lưu tham số và seed. Các biến đổi là phép thử tổng hợp, không được coi là mô phỏng vật lý chính xác. Kiểm tra bằng mắt các mức dùng trong train để tránh xóa sạch bằng chứng vật thể.

Với cùng checkpoint và trạng thái inference, lấy output dense trước NMS của A/B, giải mã hộp và chạy assigner với cùng GT. Không gọi API dự đoán hậu NMS để thay cho log positive của quá trình train. Kiểm tra riêng rằng cách trích xuất tái hiện được assigner trong trainer; ghi rõ khác biệt train/eval và BatchNorm.

Định danh điểm bằng `(feature_level, row, column)` thay vì số thứ tự phẳng không có ngữ cảnh. Với cùng hình học, các điểm tương ứng trực tiếp; nếu mở rộng sang translation phải có ánh xạ tọa độ và vùng hợp lệ.

Với mỗi vật thể i, lưu tập positive P_A(i), P_B(i), confidence đúng lớp, IoU của hộp dự đoán với GT, soft target weights và trạng thái cạnh tranh với vật thể lân cận.

Các chỉ số:

- Positive count và tỷ lệ GT không có positive, trước/sau giải quyết xung đột.
- Jaccard overlap của hai tập positive và churn = 1 − overlap, tính khi hợp không rỗng. Hai tập cùng rỗng được báo riêng là không có coverage, không gọi là ổn định tốt.
- Tỷ lệ điểm đổi GT được gán, chất lượng localization và confidence của positive được giữ/mất/thêm.
- Recall và lỗi trước/sau NMS theo size, lớp, độ tương phản và mức suy giảm.
- So sánh instability sau khi kiểm soát hoặc phân tầng kích thước, lớp, confidence, số positive. Không dùng cùng một proxy để vừa định nghĩa chất lượng vừa tuyên bố đã chứng minh detection tốt hơn.

Báo kích thước vật thể ở ảnh gốc và sau resize/letterbox. AP_small chuẩn COCO dùng protocol evaluator được công bố; nhóm tiny theo số pixel sau resize là phân tích bổ sung, phải định nghĩa riêng.

Pilot quan sát khoảng 300–500 ảnh train/validation đa dạng, nhưng không gọi đó là cỡ mẫu bảo đảm ý nghĩa thống kê. Quét suy giảm trên tập lớn hơn nếu tín hiệu ban đầu đủ rõ.

## 7. Prototype tối thiểu

Các thành phần dự kiến: paired-view transform; bộ trích xuất dense predictions; assignment logger; assigner có tham khảo view A; loss/trainer hai view; evaluator chung. Đây là các trách nhiệm logic, chỉ tách file khi triển khai thực sự cần.

V0: cùng một detector xử lý A/B, mỗi view dùng TAL độc lập. Loss là trung bình hai detection loss. Đây là baseline hai view và là đường code nền cho mọi biến thể.

V1: dùng assignment trên A làm tham khảo trực tiếp cho B. Chỉ chuyển lựa chọn trong tập candidate hợp lệ theo phiên bản assigner đã khóa; giữ GT thật làm target lớp/hộp. Đây là đối chứng đơn giản, không phải phương pháp cuối.

V2: chuyển có điều kiện. Đánh giá chất lượng A bằng confidence đúng lớp và overlap với GT; kết hợp với bằng chứng B để xếp hạng candidate. Giữ ngân sách top-k/candidate policy như baseline trong thí nghiệm chính; dùng fallback về TAL khi tham khảo không đáng tin. Chưa cố định công thức trọng số trước khi xem diagnostic report.

Quy tắc triển khai:

- Warm-up với cùng protocol ở mọi nhánh; ban đầu dùng cùng mô hình, chưa thêm EMA teacher.
- Detach tín hiệu dùng để chọn/xếp hạng assignment. Detection loss của A vẫn được học bình thường; không vô tình stop-gradient toàn nhánh A trong phương pháp mới khi baseline có gradient.
- B dùng GT thật, không dùng hộp dự đoán A làm nhãn thay thế. Có cùng normalization loss giữa các phương pháp; log thay đổi target weight và positive count để tránh gain do scale gradient.
- Giải quyết xung đột nhiều GT nhất quán; không giao một điểm cho hai vật thể. Không hứa bảo đảm một positive cho mọi GT nếu hình học/cạnh tranh không cho phép; báo các trường hợp còn thiếu coverage.
- Khi thử cơ chế mở rộng coverage, có baseline coverage-only dùng chính policy đó. Kiểm tra nhóm nhỏ/hiếm/khó không bị bỏ học vì thiếu ổn định.
- Không ép consistency khi vật thể mất hết bằng chứng. Bắt đầu với suy giảm vừa phải; phân tích sensitivity và subset được kiểm tra thủ công thay vì tự lọc mọi mẫu khó bằng confidence thấp.

Kiểm tra đúng trước khi full train: A=B tái hiện assignment nền; tắt can thiệp cho kết quả như V0; mapping tọa độ đúng; empty labels; GT chồng lấp; một điểm nhiều GT; không có candidate; loss/gradient hữu hạn; resume checkpoint; export không mang nhánh train phụ.

## 8. Ma trận thí nghiệm và ngân sách

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| E0 | YOLOv8n | Baseline gốc |
| E1 | YOLOv8n-P2 | Tác động của P2 |
| E2 | P2 + suy giảm ngẫu nhiên một view | Augmentation đơn thuần giải quyết được bao nhiêu? |
| E3 | P2 + hai view, TAL độc lập | Tác động của thêm dữ liệu/compute và hai loss |
| E4 | E3 + prediction consistency thông thường | Assignment mới có hơn consistency không? |
| E5 | E3 + chuyển assignment A sang B | Chép lựa chọn đơn giản đã đủ chưa? |
| E6 | E3 + chuyển có điều kiện | Đóng góp cơ chế đề xuất |
| E7 | E3 + coverage-only | Gain có chỉ do số/cách phủ positive không? |

E7 cần policy xác định ở bước thiết kế; nếu E6 có mở rộng coverage, thêm E6 dùng cùng policy để tách tương tác. Đối chiếu STAL hoặc phương án coverage gần nhất ở cùng kiến trúc khi tương thích; việc port phải được mô tả, không coi một bản port chưa xác minh là kết quả bài gốc.

Screening một seed và budget ngắn chỉ để chọn cấu hình. Chạy full tối thiểu ba seed cho E3, E5, E6 và đối chứng mạnh nhất từ E2/E4/E7; ghi rõ cấu hình nào được chạy nhiều seed. Cân bằng số update và số view cho so sánh chính. Với baseline một view, báo thêm thời gian/ảnh train và đối chứng train lâu hơn cùng ngân sách nếu lợi ích có thể đến từ compute.

Không quét tất cả tổ hợp siêu tham số. Đặt trước ngân sách validation cho mỗi phương pháp; giữ test cho lần đánh giá đã chốt. Mỗi run phải có run ID, model/data/code hash, config, seed, thời gian train và peak VRAM.

Đánh giá trên ít nhất hai bộ dữ liệu với model train riêng và protocol công bố. DUO dùng làm benchmark bổ sung cần ghi quan hệ nguồn với URPC; không gọi URPC→DUO là chuyển miền độc lập khi chưa loại trùng/gần trùng. Bộ khác miền như RUOD cần audit nhãn và class mapping; nhãn thiếu có thể làm sai FP. Nếu train gộp hoặc chuyển miền, kiểm tra trùng giữa toàn bộ nguồn và split.

## 9. Tiêu chí đánh giá và quyết định

Chỉ số chính: mAP50–95 và AP_small trên ảnh thực, recall vật thể nhỏ, kết quả từng lớp. Chỉ số bổ sung: AP50, AP theo nhóm suy giảm/size, robustness trên corruption giữ riêng loại hoặc cường độ chưa dùng để tuning, FP/image tại recall tương đương.

Báo mean/std qua seed; dùng paired bootstrap theo ảnh hoặc theo scene nếu có phụ thuộc ảnh để ước lượng uncertainty của chênh lệch, đồng thời báo biến thiên seed. Không đánh đồng số hộp là số mẫu độc lập. Báo kết quả âm và ví dụ gate/assignment sai, không chỉ chọn ảnh đẹp.

Tiếp tục viết claim phương pháp khi E6 vượt đối chứng mạnh về detection thực, kết quả lặp lại và diagnostic phù hợp cơ chế. Không đặt trước một ngưỡng AP tăng bảo đảm được nhận bài. Nếu gain biến mất ở đối chứng cùng compute hoặc coverage-only, không quy lợi ích cho cross-view assignment.

Latency đo trên cùng máy/runtime/input/batch, có warm-up và đồng bộ GPU. Nếu bỏ hết thành phần train phụ, graph inference có thể giữ nguyên so với YOLOv8-P2; xác minh cấu trúc, số tham số và export. Không gọi chi phí inference toàn hệ thống bằng 0.

## 10. Danh sách việc và mốc gần nhất

| ID | Việc | Ưu tiên | Ước lượng | Phụ thuộc | Hoàn thành khi |
|---|---|---|---|---|---|
| T01 | Bảng related work và ranh giới claim | P0 | 2–3 ngày | — | Có đối chiếu cơ chế, không chỉ abstract |
| T02 | Kiểm tra dataset, split và mapping | P0 | 1–2 ngày | — | Có audit/manifest, test được giữ riêng |
| T03 | Chọn phiên bản, cấu hình P2, smoke train/resume | P0 | 1–2 ngày | T02 | Chạy lại được cùng config |
| T04 | Train baseline và lưu artifacts | P0 | 2–3 ngày + GPU | T03 | Có checkpoint và metrics |
| T05 | Logger positive và ảnh trực quan A/B | P0 | 2–3 ngày | T03 | Mapping và output được đối chiếu |
| T06 | Diagnostic report H1/H2 | P0 | 2–3 ngày | T04, T05 | Có kết luận tiếp tục/đổi hướng |
| T07 | Trainer hai view và direct-copy control | P1 | 2–3 ngày | T01, T06 | E3/E5 chạy đúng và có sanity checks |
| T08 | Chuyển có điều kiện và coverage controls | P1 | 2–4 ngày | T07 | Có E6 và đối chứng tách cơ chế |
| T09 | Full evaluation, nhiều seed/dataset | P1 | 7–12 ngày + GPU | T08 | Kết quả tái lập, có uncertainty |
| T10 | Bản thảo và gói tái lập | P1 | 4–6 ngày | T09 | Mọi claim truy được về config/kết quả |

Các ước lượng ở bảng task là phần chia nhỏ của các giai đoạn, không cộng thêm lần nữa. Người thực hiện mặc định là chủ dự án; chưa có phân công cho người khác hoặc đặt lịch chạy tự động.

Mốc ưu tiên đầu tiên: hoàn thành T02–T06 và bản đối chiếu T01. Chưa bắt đầu full ablation hoặc viết phần Results khi chưa có diagnostic report. Không có task huấn luyện nào đã được thực hiện chỉ bởi việc lập tài liệu này.

Các vị trí đầu ra dự kiến khi triển khai: `configs/`, `src/`, `scripts/`, `tests/`, `reports/`. Dataset, checkpoint và run artifacts lưu ở vùng dữ liệu riêng đã được ignore; không commit chúng vào Git.

## 11. Nguồn cần đọc và trạng thái

- [TAL/TOOD, ICCV 2021](https://arxiv.org/abs/2108.07755): cơ sở task-aligned assignment; cần đọc công thức và đối chiếu implementation đã khóa.
- [Ultralytics TaskAlignedAssigner](https://docs.ultralytics.com/reference/utils/tal/), [detection loss](https://docs.ultralytics.com/reference/utils/loss/): tham khảo chính thức, tài liệu online thay đổi theo phiên bản; chưa thay cho việc kiểm tra source đã pin.
- [STAL, tài liệu tác giả](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd): đối chứng coverage; cần kiểm tra mã cụ thể trước khi port.
- [Consistent-Teacher, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.pdf): assignment inconsistency trong semi-supervised detection; cần phân biệt supervision và cơ chế.
- [DCP-UDet](https://www.preprints.org/manuscript/202610.2771): preprint ngày 28-10-2026, chưa phản biện ở bản đã kiểm tra; đã có paired-view consistency dưới suy giảm, cần đọc sâu để tránh claim trùng.
- [DUO, bài gốc](https://arxiv.org/html/2106.05681v1): nguồn dữ liệu tổng hợp và gán lại nhãn; dùng để kiểm tra quan hệ giữa benchmark.
- [Các hướng nghiên cứu đã có](research-options.md), [literature review](literature-review.md): bối cảnh trong repo, chưa phải bằng chứng rằng phương pháp hiện tại đã mới hoặc hiệu quả.
