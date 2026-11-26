# Các phương án nghiên cứu để cải tiến YOLO cho vật thể nhỏ

Tài liệu đi kèm: [80 mục nghiên cứu và điểm nổi bật](literature-review.md), [danh mục nguồn](sources.json).

<div class="no-break"></div>

## Kết luận để ra quyết định

Với trạng thái dự án hiện tại, **P2 + Gray World + CLAHE nên được dùng làm baseline**, chưa đủ làm đóng góp phương pháp chính. Nên chọn một cơ chế lỗi có bằng chứng, phát triển một giải pháp có giả thuyết rõ, rồi kiểm tra trên ít nhất hai bộ dữ liệu khác nhau.

Tám hướng dưới đây là **đề xuất nghiên cứu sau khi đối chiếu tài liệu**, chưa phải kết quả thực nghiệm và chưa được chứng minh mới tuyệt đối. Mỗi hướng có nghiên cứu gần; khác biệt phải được xác minh bằng full text và ablation. Chỉ thay phiên bản YOLO, đổi tên module, hoặc ghép nhiều block có sẵn không tự tạo thành đóng góp khoa học.

Do bạn muốn tự quyết định, bảng không giả định GPU hoặc nơi nộp bài. Mức khả thi là nhận định kỹ thuật tương đối, không phải lời hứa về thời gian chạy hoặc khả năng được chấp nhận.

## So sánh tám phương án

| Hướng | Vấn đề chính | Can thiệp | Chi phí train tương đối | Chi phí inference | Khó triển khai | Rủi ro trùng |
|---|---|---|---|---|---|---|
| A. Assignment ổn định theo bằng chứng | Tiny có giám sát không ổn định giữa các view | Assigner + consistency khi train | Vừa; cần view/teacher phụ | Có thể giữ nguyên | Vừa–cao | Cao: STAL, RFLA, SimD, TOLF |
| B. Fusion tầng nông có giám sát độ tin cậy | Feature độ phân giải cao khuếch đại cả clutter | Neck/P2 + supervision cho gate | Vừa | Tăng nhẹ–vừa, phải đo | Vừa | Rất cao: SET, ASFF, DSG-YOLO26, DM-EFS |
| C. Chọn vùng độ phân giải cao theo lợi ích và ngân sách | Tính toán bị dùng vào nền | Router + crop/feature routing | Vừa–cao | Điều khiển theo ngân sách; nhiều passes | Cao | Cao: AutoFocus, QueryDet, ESOD, SCAHI |
| D. Distillation có kiểm soát lỗi crop/teacher | Student nhỏ thiếu bằng chứng, teacher có thể dạy sai | Teacher crop + KD khi train | Cao hơn một detector; thêm teacher | Giữ nguyên nếu bỏ hết nhánh phụ | Vừa–cao | Cao: FGD, ScaleKD, CSCWD |
| E. Downsampling theo pha và nhất quán dịch | Vật thể vài pixel nhạy với vị trí lấy mẫu | Các stage đầu + equivariance loss | Vừa | Tùy số pha giữ lại | Cao | Cao: SPD-Conv, anti-aliasing, APS |
| F. Giữ hai vật thể sát nhau qua toàn pipeline | Mẫu dương, neck hoặc hậu xử lý nhập hai mục tiêu | Pair-aware training / matching | Vừa | Có thể gần nguyên | Cao | Cao: density-aware NMS, NMS-free matching |
| G. Enhancement/SR chọn lọc phục vụ detection | Khôi phục ảnh có thể thêm artifact/hallucination | Module chọn lọc hoặc train-only guidance | Vừa–rất cao tùy SR | Tăng nếu còn module lúc test | Vừa–cao | Rất cao: Perceptual GAN, UnitModule, DPNet, SET |
| H. Benchmark độ bền và nhãn nhiễu kèm cơ chế sửa đơn giản | Tăng AP có thể chỉ do split, resize hoặc label artifacts | Protocol + corruption/noise-aware training | Ít sửa code nhưng nhiều lần đánh giá | Có thể giữ nguyên | Vừa về code, cao về nghiên cứu | Cao: AI-TOD-v2, DN-TOD, TOLF, nghiên cứu UIE |

“Giữ nguyên inference” chỉ đúng nếu implementation bỏ toàn bộ teacher, auxiliary heads và nhánh kiểm tra khi export. Mọi tỉ lệ chi phí cần đo từ baseline thật; không suy FPS từ FLOPs.

**Ưu tiên cân bằng để khảo sát:** A hoặc D nếu muốn giữ kiến trúc triển khai gọn; C nếu muốn câu chuyện accuracy–compute rõ; E nếu muốn đào sâu cơ chế sampling. B dễ bắt đầu từ repo P2 nhưng đông công trình gần, cần kiểm tra R57 trước. H phù hợp khi muốn một bài phân tích thực nghiệm chắc thay vì chạy theo tên module.

## A. Assignment ổn định theo bằng chứng giữa nhiều view

**Câu hỏi:** Vật thể tiny có bị đổi hoặc mất positive assignments khi ảnh chỉ dịch 1–2 pixel hay suy giảm nhẹ, và điều này có giải thích được lỗi recall của YOLO?

**Giả thuyết:** Một phần thất bại đến từ supervision không ổn định theo bằng chứng ảnh, ngoài việc thiếu candidate hình học. Dùng độ ổn định sau khi căn tọa độ để điều chỉnh trọng số học có thể tốt hơn chỉ tăng số positive.

**Thiết kế tối thiểu:**

1. Dùng hai view có biến đổi tọa độ đã biết; trước hết chỉ translation nhẹ và photometric noise kiểm soát. Gán candidate về cùng GT trong train, đưa prediction về cùng hệ tọa độ.
2. Đo disagreement chuẩn hóa theo kích thước vật thể, cùng agreement về lớp. Nguồn có thể là EMA teacher; stop-gradient nhánh tính reliability để hạn chế tự giảm loss bằng cách tăng uncertainty.
3. Dùng reliability như trọng số mềm trong chọn mẫu/loss, kèm warm-up. Có thể thử score dạng `a_ij = s_ij^alpha * overlap_ij^beta * exp(-u_ij / tau)`; đây chỉ là công thức khởi đầu cần kiểm chứng.
4. Giữ một cơ chế positive coverage giống nhau ở baseline và phương án mới. Đặt sàn trọng số để không bỏ luôn tiny/rare classes khó. Mọi candidate expansion phải đối chiếu STAL.

**Phần cần chứng minh khác biệt:** Độ bất ổn *gắn với biến đổi ảnh và assignment* mang thông tin hữu ích ngoài confidence, kích thước, IoU, static Gaussian và uncertainty định vị. Không tuyên bố “uncertainty-aware loss mới” nếu chỉ giống TOLF.

**Đối chứng gần:** [TOOD](https://arxiv.org/abs/2108.07755), [RFLA](https://arxiv.org/abs/2208.08738), [SimD](https://arxiv.org/abs/2407.02394), [STAL trong YOLO26](https://arxiv.org/abs/2606.03748), [TOLF](https://arxiv.org/abs/2601.00617).

**Ablation:** coverage only; confidence weighting; size weighting; reliability không consistency; consistency không sửa assignment; kết hợp. Báo positive-count và assignment-flip-rate theo size, nhưng đánh giá chính vẫn AP/AR.

**Rủi ro:** Hai view cùng sai vẫn có thể rất ổn định; bootstrap teacher kém sẽ loại mẫu khó. Kiểm tra correlation reliability–độ đúng bằng GT validation, rare-class recall và cảnh isolated tiny. Dừng nếu chỉ cải thiện training loss hoặc lợi ích biến mất khi baseline dùng STAL/augmentation tốt.

**Khả thi:** Sửa trainer/assigner cần hiểu rõ thư viện nhưng không phải dựng backbone mới. Tốn train thêm do view/teacher; inference có thể hoàn toàn giữ nguyên.

**Tên paper dự kiến:** “Evidence-Stable Assignment for Tiny Object Detection”. Chỉ dùng như tên làm việc.

## B. Fusion tầng nông có giám sát độ tin cậy của chi tiết

**Câu hỏi:** P2 tăng recall nhờ chi tiết foreground, nhưng có tăng false positives vì truyền nền tần số cao, noise hoặc texture lặp lại?

**Giả thuyết:** Fusion nên chọn chi tiết có ích theo vùng, thay vì truyền toàn bộ high-resolution feature hay khuếch đại mọi high-frequency component.

**Thiết kế tối thiểu:** Giữ một nhánh detail tầng nông và một nhánh semantic ở cùng resolution; gate không gian quyết định lượng residual detail được thêm vào. Khởi đầu bằng convolution nhẹ. Dùng GT-derived foreground/background masks *chỉ trong train* để kiểm soát amplification; thêm giám sát qua các mức degradation, tránh gate học đơn thuần “cạnh mạnh = đối tượng”. Chỉ thử wavelet nếu baseline cho thấy thành phần phổ giúp dự báo đúng/sai.

**Phần cần chứng minh khác biệt:** Gate được hiệu chỉnh bởi lợi ích detection hoặc độ bền của foreground có tốt hơn gate generic, spatial attention, semantic admission và smoothing nền? Bản thân dual-stream/gate/wavelet chưa phải đóng góp mới.

**Đối chứng gần:** [ASFF](https://arxiv.org/abs/1911.09516), [SET](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html), [DM-EFS](https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html), [DSG-YOLO26](https://www.sciencedirect.com/science/article/pii/S0923596526002006), và ReDHF (R56). R57 ở mức S nên chưa thể tuyên bố gate đề xuất khác bài đó.

**Ablation:** P2 baseline; concat/add cùng kênh; generic attention; gate không supervision; gate có supervision; foreground-only vs có hard background. Giữ gần bằng params/compute bằng baseline tăng kênh hoặc thêm convolution đơn giản.

**Rủi ro:** Gate có thể chỉ mã hóa size, làm mất vật thể tương phản yếu hoặc làm tăng compute quá lợi ích. Dừng nếu AP tăng tương tự khi chỉ tăng số kênh hoặc lợi ích chỉ đến từ P2.

**Khả thi:** Là hướng gần nhất với repo hiện tại, có thể phát triển theo từng stage. Chi phí inference phải bao gồm toàn bộ P2 head, không chỉ gate. Khả năng tạo bài phụ thuộc việc chỉ ra cơ chế tốt hơn các tiền lệ rất gần.

**Tên làm việc:** “Reliability-Supervised Shallow Fusion for Small Object Detection”.

## C. Phân bổ độ phân giải theo lợi ích kỳ vọng và ngân sách

**Câu hỏi:** Với cùng latency hoặc cùng số pixel được xử lý, nên dành high resolution cho vùng nào để giảm bỏ sót nhiều nhất?

**Giả thuyết:** Objectness/density chưa đủ: vùng thưa có một tiny object khó vẫn đáng được phóng to. Ước lượng lợi ích của một lượt xử lý bổ sung có thể tốt hơn chọn vùng đông hoặc chia đều.

**Thiết kế tối thiểu:**

- Một lượt global thấp hơn tạo feature dùng chung; router nhẹ dự đoán utility của các vùng.
- Tạo target utility trên train từ chênh lệch matched recall hoặc detection loss giữa coarse và một lượt high-resolution. Đây là proxy cục bộ, không gọi là gradient trực tiếp của dataset AP.
- Chọn top-k vùng dưới ngân sách cố định; có overlap/coverage constraint và một phần khám phá để tránh bỏ vùng isolated. Bắt đầu bằng top-k/greedy, chưa cần reinforcement learning.
- Ghép kết quả phải kiểm soát vật thể cắt biên và vùng phủ. Nếu dùng dense fusion cần so với SCAHI; nếu dùng feature slicing cần so với ESOD.

**Phần cần chứng minh khác biệt:** Utility được hiệu chỉnh theo rủi ro bỏ sót và chi phí có vượt density routing, uncertainty routing và crop cố định không? Novelty nằm ở mục tiêu phân bổ và cơ chế bảo toàn recall, không nằm ở chữ “adaptive crop”.

**Đối chứng:** [AutoFocus](https://arxiv.org/abs/1812.01600), [QueryDet](https://arxiv.org/abs/2103.09136), [SAHI](https://arxiv.org/abs/2202.06934), [ESOD](https://arxiv.org/abs/2407.16424), [SCAHI](https://arxiv.org/abs/2611.12657); thêm ASAHI/DMNet ở R45/R41.

**Ablation:** uniform grid; random k crops; density top-k; confidence/entropy top-k; utility top-k; có/không exploration; cùng passes nhưng thay merge. So single-pass input lớn ở cùng latency trước tiên.

**Đo bắt buộc:** APtiny–latency curve; latency p50/p95; end-to-end preprocessing + router + copies + detector + merge; recall vùng chọn cho tiny đứng riêng và theo mật độ; peak VRAM.

**Rủi ro:** Router bỏ sót thì detector sau không cứu được; sparse FLOPs có thể không nhanh vì crop/copy và kernel overhead. Nếu cùng latency single-pass resolution cao vẫn thắng, không nên tiếp tục một router phức tạp chỉ vì AP cao hơn baseline 640.

**Khả thi:** Khó hơn A/D về pipeline và đo tốc độ; phù hợp dữ liệu ảnh lớn như VisDrone/SODA-A/DOTA. DOTA phải tách nhãn hộp ngang/xoay. Có thể là bài về accuracy–compute rõ nếu lợi ích bền trên nhiều ngân sách.

**Tên làm việc:** “Miss-Risk-Aware Resolution Allocation for Small Object Detection”.

## D. Distillation theo độ tin cậy của teacher và ngữ cảnh crop

**Câu hỏi:** Teacher nhìn crop phóng to giúp tiny object, nhưng crop thiếu context hoặc cắt vật thể có truyền tri thức sai cho student nhìn toàn ảnh không?

**Giả thuyết:** Tín hiệu teacher nên được chọn theo tính đúng, độ ổn định và nguồn crop, thay vì distill mọi feature high-resolution với trọng số như nhau.

**Thiết kế tối thiểu:** Train một teacher đủ mạnh trên cùng train split; teacher nhận crop ở scale phù hợp, student nhận toàn ảnh. Căn chỉnh chính xác feature/predictions về hệ tọa độ student; mask vùng crop bị cắt cụt hoặc thiếu context. Kết hợp matched-GT quality trong train và agreement giữa crop đủ context với crop sát để điều chỉnh KD. GT chỉ có ở train, không ở test. Student inference giữ nguyên.

**Phần cần chứng minh khác biệt:** Cách xử lý lỗi do crop/context và teacher reliability phải tốt hơn KD cùng scale, ScaleKD, FGD và P2-to-P3 CSCWD. Chỉ đổi teacher YOLO11m-P2 thành YOLOv8m-P2 không mới.

**Đối chứng:** [FGD](https://arxiv.org/abs/2111.11837), [ScaleKD](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html), [CSCWD](https://arxiv.org/abs/2609.30395), CFINet (R50).

**Ablation:** không KD; teacher full-image; teacher crop + KD thường; alignment; truncation mask; confidence-only weight; reliability/context weight; teacher cùng resolution. Báo teacher đúng/sai trên những GT mà student được hướng dẫn.

**Rủi ro:** Student không có đủ pixel để tái tạo biểu diễn teacher; alignment sai có thể tạo lợi ích giả hoặc phá training. Tăng input student là baseline bắt buộc. Cache teacher chỉ hợp lệ nếu augmentation và phép căn chỉnh vẫn đúng; không coi caching là miễn phí khi đổi crop mỗi epoch.

**Khả thi:** Chi phí train tăng và cần teacher; có thể giảm peak VRAM bằng frozen teacher chạy tuần tự nhưng sẽ đổi tốc độ. Điểm mạnh là inference không tăng nếu export đúng. Đây là lựa chọn hợp lý khi ưu tiên triển khai gọn, nhưng cần đóng góp vượt reliability-weighted KD thông thường.

**Tên làm việc:** “Context-Validated Cross-Scale Distillation for Tiny Object Detection”.

## E. Downsampling theo pha và nhất quán định vị khi dịch nhỏ

**Câu hỏi:** Với vật thể vài pixel, chỉ dịch ảnh một pixel có làm đối tượng rơi vào pha sampling bất lợi và biến mất khỏi feature không?

**Giả thuyết:** Giữ hoặc phối hợp có kiểm soát các pha lấy mẫu ở tầng đầu, kèm giám sát equivariance của hộp, có thể tăng recall tiny mà ít cần tăng toàn bộ input resolution.

**Thiết kế tối thiểu:** Chỉ thay một hoặc hai stage đầu. So strided conv với space-to-depth và anti-aliased downsampling; nếu có bằng chứng mới thiết kế phase mixture chọn lọc. Khi train, dịch ảnh bằng phép nguyên pixel có tọa độ chính xác và penalize sai khác prediction sau khi đưa ngược về vị trí gốc.

**Phần cần chứng minh khác biệt:** Lựa chọn pha cho detection phải bảo toàn *equivariance*: ảnh dịch thì hộp dịch theo. Không yêu cầu đầu ra bất biến vị trí như classification. Cần cơ chế hoặc phân tích mới hơn SPD-Conv và APS.

**Đối chứng:** [SPD-Conv](https://arxiv.org/abs/2208.03641), [anti-aliased CNNs](https://arxiv.org/abs/1904.11486), [Adaptive Polyphase Sampling](https://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html), thêm resize/translation augmentation và P2 thường.

**Ablation:** sampling change only; loss only; kết hợp; giữ một pha vs mọi pha; cùng channel budget. Test các offset chưa dùng trong train; mask vùng rìa và dùng padding nhất quán.

**Rủi ro:** Anti-aliasing có thể xóa tiny signals; chọn pha theo năng lượng toàn ảnh có thể ưu tiên nền. Resize/interpolation và padding là confounder lớn. Dừng nếu lợi ích biến mất khi kiểm soát augmentation hoặc chỉ tốt ở một offset.

**Khả thi:** Có câu hỏi cơ chế rõ, nhưng khó phân tích và dễ sai tọa độ. Hợp khi muốn nghiên cứu kỹ thuật sâu hơn ghép module.

**Tên làm việc:** “Phase-Consistent Sampling for Tiny Object Localization”.

## F. Giữ hai vật thể nhỏ sát nhau qua assignment và dự đoán

**Câu hỏi:** Khi hai tiny GT rất gần nhau, chúng bị nhập từ candidate assignment, feature fusion, hay NMS/giới hạn số detections?

**Giả thuyết:** Bảo toàn riêng từng instance sát nhau trong training có thể quan trọng hơn tăng ngữ cảnh chung, đặc biệt ở ảnh mật độ cao.

**Thiết kế tối thiểu:** Trước tiên log lỗi theo khoảng cách tâm chuẩn hóa và overlap GT. Xác định giai đoạn làm mất instance. Chỉ khi lỗi nằm ở training mới thêm pair-preservation supervision: hai GT gần cần có đại diện dương và prediction riêng, đồng thời không làm tăng duplicate trên cùng một GT. Không đổi NMS nếu baseline NMS-free đã giải quyết phần lớn lỗi.

**Phần cần chứng minh khác biệt:** Cơ chế phải vượt one-to-one matching, density-aware suppression và loss tách đối tượng đã có trong nghiên cứu crowd. “Repulsion” hoặc NMS thích nghi là hướng cũ; cần rà thêm tài liệu crowd/repulsion nếu chọn đề tài này.

**Đối chứng đã xác minh:** [Soft-NMS](https://arxiv.org/abs/1704.04503), [Adaptive NMS](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.html), [YOLOv10](https://arxiv.org/abs/2405.14458), [YOLO26](https://arxiv.org/abs/2606.03748), [DQ-DETR](https://arxiv.org/abs/2404.03507), [DEIM](https://arxiv.org/abs/2412.04234).

**Ablation:** cùng NMS nhưng thay train; cùng train nhưng thay NMS; NMS-free; tăng max_det đồng đều; pair supervision vs chỉ tăng positives. Đo pair recall theo khoảng cách, duplicate rate và AP toàn bộ.

**Rủi ro:** GT không hoàn chỉnh làm pair loss dạy sai; ép tách có thể chia một vật thể thành nhiều hộp. Đây là hướng có nhiều tiền lệ ngoài YOLO, cần tìm sâu thêm trước khi chốt novelty.

**Khả thi:** Tốn phân tích lỗi và sửa assigner/matching, inference có thể ít đổi. Hợp VisDrone, AI-TOD-v2, SODA-A; không phù hợp nếu lỗi chủ yếu là vật thể quá mờ đứng riêng.

**Tên làm việc:** “Instance-Preserving Learning for Crowded Tiny Object Detection”.

## G. Enhancement hoặc super-resolution chọn lọc theo lợi ích detection

**Câu hỏi:** Khi nào enhancement/SR giúp tiny object, và khi nào nó tăng texture giả hoặc làm dịch phân bố ảnh?

**Giả thuyết:** Chỉ áp dụng khôi phục ở vùng có bằng chứng đáng tin và hiệu quả detection kiểm chứng được sẽ tốt hơn xử lý cố định toàn dataset.

**Thiết kế tối thiểu:** Bắt đầu từ raw vs simple enhancement với cùng detector. Dùng selector học trên train để chọn raw/enhanced hoặc residual nhẹ theo vùng; target dựa trên detection supervision trong train. Nếu chuyển sang SR, tách dữ liệu suy giảm tổng hợp và ảnh low-resolution thực; cần kiểm tra false positives trên background khó. Có thể dùng enhanced branch chỉ làm teacher khi train để tránh tăng inference.

**Phần cần chứng minh khác biệt:** Selective enhancement đã được gợi ý rõ và joint enhancement đã tồn tại. Cần một cách ước lượng lợi ích triển khai được, không dùng GT ở test, và bằng chứng tránh artifact/hallucination. Không gọi oracle mixed-set upper bound là model.

**Đối chứng:** [Perceptual GAN](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.html), [UnitModule](https://arxiv.org/abs/2309.04708), [DPNet](https://arxiv.org/abs/2307.03536), [Is UIE All Detectors Need?](https://arxiv.org/abs/2311.18814), [Revisiting UIE](https://www.mdpi.com/2313-433X/12/1/18), SET (R51). Với general SOD, phải bổ sung tìm kiếm SR chuyên sâu trước khi chốt.

**Ablation:** raw; enhanced fixed; raw/enhanced mixed training; selector giản đơn theo contrast; selector học; module cùng compute; train-only teacher. Đánh giá cả ảnh clean để phát hiện negative transfer.

**Rủi ro:** PSNR/SSIM/UCIQE tốt nhưng AP kém; synthetic downsampling không đại diện low-resolution thực; SR sinh chi tiết không có thật. Mọi metric hình ảnh là phụ, AP và hallucinated detections mới quyết định.

**Khả thi:** Enhancement nhẹ dễ thử với repo hiện tại; SR/diffusion nặng và khó tạo đối chứng công bằng. Với mục tiêu bài đầu tiên, nên thăm dò baseline trước khi đầu tư model khôi phục lớn.

**Tên làm việc:** “Detection-Utility-Guided Selective Restoration for Small Objects”.

## H. Nghiên cứu độ bền theo kích thước và chất lượng nhãn

**Câu hỏi:** Một cải tiến YOLO còn hiệu quả khi thay scale, blur, noise, nén ảnh, annotation jitter và dataset không?

**Giả thuyết:** Những cải tiến có AP clean tương tự có thể khác mạnh về failure profile; một regularizer nhỏ theo nhóm size/degradation có thể giúp tổng quát tốt hơn thêm module.

**Thiết kế tối thiểu:** Xây protocol có seed cố định, corruption severity kiểm soát, noise chỉ tiêm vào train khi nghiên cứu nhãn nhiễu, test clean làm tham chiếu. Đánh giá vài YOLO đại diện và P2, rồi mới đề xuất một cơ chế sửa nhẹ dựa trên lỗi đã quan sát.

**Phần cần chứng minh khác biệt:** Danh sách benchmark đơn thuần chưa đủ. Cần phát hiện mới có thể lặp lại, kiểm soát confounder, protocol hoặc tài nguyên hữu ích và phân tích giải thích được. Không tái đặt tên thí nghiệm đã có trong AI-TOD-v2, DN-TOD hoặc TOLF.

**Đối chứng:** [NWD-RKA/AI-TOD-v2](https://arxiv.org/abs/2206.13996), [DN-TOD](https://arxiv.org/abs/2401.08056), [TOLF](https://arxiv.org/abs/2601.00617), [nghiên cứu nhãn thiếu WACV 2025](https://openaccess.thecvf.com/content/WACV2025W/MaCVi/html/Lucas_Underwater_Image_Enhancement_and_Object_Detection_Are_Poor_Object_Detection_WACVW_2025_paper.html).

**Ablation:** cùng weights trên clean/corruptions; matched augmentation; noise-aware method vs training lâu hơn; clean vs noisy train labels; size bins và class. Giữ test annotations cố định. Nếu có audit nhãn, dùng quy trình độc lập và phát hành phiên bản rõ ràng.

**Rủi ro:** Cần nhiều lượt chạy dù ít sửa model; không được tối ưu trực tiếp trên corruption test. Chuyển miền giữa dataset khác lớp hoặc nguồn ảnh chồng nhau tạo kết luận sai nếu không kiểm tra.

**Khả thi:** Khởi đầu được bằng nano models và một dataset; mở rộng kiểm chứng sau. Có thể phù hợp bài thực nghiệm/benchmark, không mặc định dễ đăng hơn paper phương pháp.

**Tên làm việc:** “When Do Small-Object YOLO Improvements Generalize?”

## Thiết kế thí nghiệm dùng chung

### 1. Dữ liệu và phạm vi tuyên bố

- **Khởi đầu:** VisDrone DET theo một phiên bản/split chính thức cố định, vì gần bài toán YOLO nhỏ và đa lớp.
- **Tiny chuyên biệt:** AI-TOD-v2 để kiểm tra các đối tượng rất ít pixel, dùng evaluator và scale bins gốc.
- **Khác miền:** SODA-D; SODA-A nếu nghiên cứu ảnh trên không độ phân giải rất cao.
- **Mở rộng từ dự án:** URPC2020, DUO hoặc RUOD. Kiểm tra trùng ảnh/nguồn trước khi dùng để chứng minh generalization. Nếu class spaces khác, ánh xạ các lớp chung và báo rõ thay vì tính mAP không tương thích.
- DOTA phù hợp slicing/ảnh lớn, nhưng OBB và HBB phải là các thí nghiệm được ghi rõ riêng.
- COCO là phép kiểm tra general-object và AP small bổ sung khi đủ ngân sách, không bắt buộc là dataset đầu tiên.
- Nếu chỉ có nhãn validation công khai, báo kết quả trên validation; không đổi tên thành test.

### 2. Các baseline cần làm trước

| Baseline | Mục đích |
|---|---|
| YOLOv8n/s chuẩn, ảnh raw | Mốc khớp repo hiện có |
| YOLOv8 tương ứng có P2 | Tách tác dụng head độ phân giải cao |
| Cùng model với training/augmentation tốt hơn | Loại khả năng module mới chỉ bù recipe yếu |
| Cùng model ở input resolution lớn hơn | Tách lợi ích pixel và kiến trúc |
| YOLO11 hoặc YOLO26 có nguồn cấu hình rõ | Tránh chỉ so với baseline cũ; YOLO26 quan trọng nếu sửa assignment |
| SAHI hoặc một baseline high-resolution phù hợp | Bắt buộc nếu nghiên cứu tiny trong ảnh lớn |
| Một đối chứng sát hướng chọn | Ví dụ CSCWD cho KD, ESOD cho routing, SET cho frequency |
| Một detector ngoài YOLO khi đủ ngân sách | RT-DETR/DQ-DETR hoặc detector phù hợp giúp kiểm tra tính tổng quát |

Không cần train toàn bộ 80 paper. Chọn 2–3 đối thủ gần nhất có code/điều kiện tái hiện phù hợp. Nếu chỉ trích kết quả trong paper, đặt bảng tham khảo riêng và ghi “khác protocol”; không xếp chung như phép so sánh công bằng.

### 3. Kiểm soát các biến dễ làm AP tăng giả

- Cùng train/val/test, class mapping, evaluator, source resolution, resize/letterbox và ignore regions.
- Cùng nguồn pretraining khi ablation; ghi rõ COCO, Objects365 hoặc scratch. YOLO26 báo recipe dùng Objects365 nên so checkpoint khác dữ liệu cần tách khỏi claim về riêng module.
- Cùng optimizer/schedule hoặc giải thích recipe riêng; báo số steps, batch hiệu dụng, augmentation, close_mosaic và early stopping.
- Báo cả fixed-epoch và training-cost khi claim hội tụ nhanh. Không so teacher có dữ liệu bổ sung với student baseline thiếu dữ liệu mà bỏ qua confounder.
- Candidate mask, positive coverage, max_det, NMS/merge thresholds và class-aware/agnostic phải được cố định bằng validation.
- Nếu scene có nhiều GT hơn max_det/evaluator maxDets, ghi rõ giới hạn và báo cùng cấu hình cho tất cả model; không tự thay metric chính thức mà không nêu.
- TTA, ensemble, extra crops và postprocessing phải có nhãn rõ và được tính vào latency.
- Chọn hướng/hyperparameters trên train/validation, khóa cấu hình trước khi dùng test. Không chọn seed đẹp nhất.

### 4. Bộ số đo

**Chính:** AP50:95, AP50, AP tiny/small theo evaluator gốc, AR tiny/small, AP theo lớp, tham số, FLOPs, peak VRAM và latency end-to-end.

**Chẩn đoán:** recall theo mật độ/khoảng cách, false positives trên nền, assignment-flip-rate, số positive/GT, lỗi classification/localization, crop-border errors và độ nhạy theo degradation.

COCO AP small thường dùng diện tích GT trong hệ tọa độ gốc nhỏ hơn (32^2); benchmark khác có định nghĩa riêng. Không trộn threshold của AI-TOD với COCO. Nếu bổ sung các dải `sqrt(w*h) < 8`, `[8,16)`, `[16,32)` tại input detector, đặt tên là **diagnostic bins tùy chọn**, ghi rõ sau resize và không gọi đó là official benchmark AP.

Độ chính xác chính nên báo mean ± std qua ít nhất 3 seeds ở các cấu hình cuối nếu khả thi. Với dữ liệu chuỗi/cảnh tương quan, dùng bootstrap theo scene/sequence khi làm khoảng tin cậy, thay vì giả định các frame độc lập. Std qua seeds và uncertainty do mẫu test là hai nguồn khác nhau.

Latency cần ghi GPU/CPU, backend, precision, batch size, resolution, warm-up, synchronization và phần pipeline được tính. Params/FLOPs thấp hơn không đủ để khẳng định nhanh hơn.

### 5. Các bước thực hiện và tiêu chí dừng

1. **Sửa nền tái lập:** bổ sung cấu hình P2 còn thiếu, thống nhất đường dẫn, pin phiên bản, lưu split và manifest. Chưa cần viết module mới.
2. **Lập baseline:** chạy raw YOLOv8, P2 và input lớn hơn trên cùng một dataset; lưu prediction để phân tích lỗi.
3. **Chẩn đoán cơ chế:** chọn một hướng A–H dựa trên lỗi quan sát, không chọn module trước rồi tìm lý do sau.
4. **Pilot:** một model nhỏ, một dataset, cùng seed/recipe để kiểm tra giả thuyết. Pilot chỉ sàng lọc; không là bằng chứng cuối.
5. **Đối chứng gần:** tái hiện ít nhất một công trình sát ý tưởng hoặc thành phần cốt lõi; đọc đầy đủ các nguồn S liên quan trước khi chốt novelty.
6. **Ablation:** tách module, supervision/loss, và training policy. Nếu ba yếu tố thì có thể dùng thiết kế (2^3) khi ngân sách cho phép; không chạy sweep khổng lồ thiếu câu hỏi.
7. **Kiểm chứng:** các cấu hình cuối nhiều seeds, ít nhất hai dataset có đặc điểm khác, một bài kiểm tra robustness/transfer phù hợp.
8. **Khóa test và viết:** chỉ viết contributions theo kết quả đã có; công khai cả điểm thất bại và điều kiện không có lợi.

Chọn trước ngưỡng lợi ích thực dụng với ngân sách của bạn. Ví dụ, “AP tiny tăng vượt biến thiên seed trong khi latency tăng không quá một mức đã định” là tiêu chí có thể kiểm chứng; một con số như +1 AP chỉ là ngưỡng dự kiến, không phải cam kết hay điều kiện đủ để đăng báo.

Dừng hoặc thu hẹp claim khi: chỉ hơn scratch baseline yếu; lợi ích mất khi tăng resolution/recipe; chỉ một lớp/seed/dataset được lợi; chi phí vượt baseline đơn giản ở cùng latency; hoặc phương pháp gần đã có cùng cơ chế và kết quả.

## Liên hệ với repo hiện tại

Đã kiểm tra lại các tệp: `train.py`, `requirements.txt`, `README.md`, `preprocessing_data/clahe.py`, `preprocessing_data/create_yaml.py`, `preprocessing_data/scripts.sh`.

- `train.py` tham chiếu `yolov8n_p2.yaml`, nhưng tệp đó chưa có trong workspace được kiểm tra.
- Script khởi tạo từ YAML, chưa có lệnh chuyển trọng số pretrained tường minh; khi tạo baseline phải quyết định và ghi rõ scratch hay pretrained.
- Gray World + CLAHE là preprocessing cố định; cần thêm baseline raw và tách từng bước để biết bước nào giúp.
- Hướng dẫn chạy shell từ thư mục con và các đường dẫn Python từ root chưa thống nhất.
- Chưa có dữ liệu, weights, predictions hoặc kết quả benchmark trong workspace để chẩn đoán lỗi thật.
- Chưa có pipeline đánh giá size bins, nhiều seeds, ablation hoặc đo latency end-to-end.

Các điểm này là việc chuẩn bị thực nghiệm, **không được tính là đóng góp khoa học**. Lần tổng hợp này không sửa mã huấn luyện, không tải dataset và không chạy training.
<div class="no-break"></div>

## Khung paper sau khi có kết quả

1. **Problem:** một failure cụ thể, được định lượng theo kích thước, điều kiện và giai đoạn pipeline.
2. **Related Work:** nhóm đúng cơ chế và so trực tiếp 3–5 bài gần nhất, gồm nghiên cứu mới 2026 nếu liên quan.
3. **Method:** một ý tưởng trung tâm, mô tả toán học/thuật toán đủ tái hiện; phân biệt thành phần kế thừa và phần mới.
4. **Experiments:** protocol, controlled baselines, chi phí, ablation, nhiều seeds và kiểm chứng khác dữ liệu.
5. **Analysis:** chứng minh cơ chế dự kiến có xảy ra; nêu điều kiện thất bại, không chỉ visualization đẹp.
6. **Reproducibility:** code, configs, seeds, checkpoints khi được phép, dataset version và evaluator.
7. **Conclusion:** chỉ tuyên bố phạm vi đã được dữ liệu chứng minh.

Không nên ghi trước “state of the art”, “novel”, “real time” hoặc một mức tăng AP cụ thể trong abstract khi chưa có bằng chứng tương ứng.
