# SpeechToInfomation Upstream Review

## Kết luận ngắn

- `D:\Workspace\SpeechToInfomation` là một sản phẩm bao ngoài lớn hơn `cherry_core`, nhưng không phải nền tốt hơn để bê nguyên vào.
- Giá trị thực tế nằm ở:
  - workflow nhiều bước cho người dùng,
  - prompt điều tra dạng plain text,
  - một số heuristic offline để trích xuất entity/timeline.
- Không nên nhập trực tiếp:
  - `Celery/Redis/Postgres`,
  - `React/MUI` monolith,
  - `Ollama/network calls`,
  - các endpoint gắn chặt với DB/task queue,
  - toàn bộ thư mục runtime/artifact/log/documentation rời rạc.

## Phần đã học chọn lọc vào `cherry_core`

### 1. Prompt và cấu trúc báo cáo

- Học từ hướng trình bày plain-text của upstream để bổ sung template `forensic_brief.j2`.
- Không dùng lại prompt token lỗi hoặc chat-format cứng của upstream.
- Giữ `deep_investigation.j2` làm prompt JSON chính cho pipeline.

### 2. Intelligence presentation offline

- Học từ ý tưởng `AnalysisPanel`, `VisualizationPanel`, `InvestigationSummaryCard`:
  - gom thông tin thành các nhóm dễ khai thác,
  - hiển thị timeline,
  - tách cảnh báo/rủi ro khỏi phần summary dài.
- Không copy code frontend React.
- Thay vào đó, dựng `IntelPresentationService` để sinh:
  - `intel_cards`
  - `intel_timeline`
  - `risk_flags`

### 3. Regex fallback tiếng Việt

- Học từ `visualization_service.py` của upstream:
  - regex phone,
  - regex time,
  - regex location,
  - nhận diện từ lóng phổ biến.
- Chỉ giữ bản tối giản, offline, không phụ thuộc service mạng.

## Phần cố ý không nhập

- Hạ tầng hàng đợi tác vụ và persistence nặng.
- `summary_service_v2.py` phụ thuộc `Ollama`.
- `transcribe_service_v2.py` kiểu tích hợp lẫn endpoint/service/DB.
- `cherry_transcription_service.py` do phụ thuộc singleton và merge logic speaker thô.
- Mọi script vận hành tạm, log, file báo cáo phiên làm việc, `node_modules`, `venv`, `storage`.

## Lý do giữ `cherry_core` làm nhánh chính

- Repo hiện tại nhẹ hơn, sạch hơn và dễ tách Git hơn.
- Luồng webapp offline-first hiện tại phù hợp hơn với yêu cầu local-only.
- Kiến trúc `application / infrastructure / presentation` rõ ràng hơn repo upstream.
- Có thể tiếp tục mở rộng intelligence UI mà không kéo theo nợ hạ tầng.
