# Hệ thống trinh sát âm thanh

Webapp và pipeline STT tiếng Việt chạy hoàn toàn cục bộ cho bài toán bóc tách hội thoại, giảm lỗi transcript, tách người nói và tạo tóm tắt nghiệp vụ điều tra/trinh sát bằng LLM local.

## Mục tiêu

- Chạy `offline-first`, không phụ thuộc dịch vụ Internet khi vận hành.
- Giữ `PhoWhisper` làm ASR mặc định cho tiếng Việt.
- Dùng `whisper-v2` trên `faster-whisper/CTranslate2` để tránh lỗi `openai-whisper` với `NumPy 2.4`.
- Cho phép bật từng bước xử lý thay vì ép chạy toàn bộ pipeline.
- Ưu tiên `GPU` khi có CUDA, tự rơi về `CPU` khi không khả dụng hoặc khi người vận hành chọn thủ công.

## Tính năng chính

- Phiên âm với `PhoWhisper`, `whisper-v2`, hoặc `WhisperX`.
- Lọc khoảng lặng bằng `Silero VAD` nếu model local đã sẵn sàng.
- Ổn định transcript và timestamp bằng `Stable-TS` theo chế độ tùy chọn.
- Giảm ảo giác ngữ cảnh và chuẩn hóa câu chữ theo từng lớp xử lý tùy chọn.
- Tách người nói offline bằng `SpeechBrain`.
- Tạo `Tóm tắt trinh sát` bằng LLM local `vistral`.
- Xuất artifact theo từng job để benchmark và đối chiếu.

## Kiến trúc chạy offline

- Webapp: `FastAPI` + giao diện tĩnh tại `presentation/web/`
- Job runner: `application/services/web_job_manager.py`
- Pipeline: `application/services/stt_web_pipeline.py`
- ASR chính:
  - `phowhisper`: tiếng Việt mặc định
  - `whisper-v2`: chạy trên `faster-whisper`, không dùng `openai-whisper`
  - `whisperx`: tùy chọn mở rộng alignment/timeline
- `Stable-TS` được vendor trong `.vendor/stable_whisper` và chỉ chạy khi bật thủ công
- `pyannote` không nằm trong luồng webapp mặc định vì không phù hợp mục tiêu offline/local-only

## Yêu cầu môi trường

- Python 3.11 trở lên
- `ffmpeg` có sẵn trong `PATH`
- Model phải có sẵn cục bộ trong thư mục `models/`
- Không cần Internet khi chạy suy luận, nhưng một số model phải được tải về trước bằng quy trình riêng

## Cấu trúc model local kỳ vọng

Các đường dẫn chính được khai báo ở [`core/config.py`](core/config.py).

- `models/phowhisper-safe`
- `models/whisper-large-v2` hoặc `models/faster-whisper-large-v2` dạng CTranslate2
- `models/silero`
- `models/speechbrain`
- `models/protonx`
- `models/vistral/vistral-7b-chat-Q4_K_M.gguf`

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` là đường cài đặt mặc định cho lần chạy đầu tiên và không bao gồm `vllm`.

Nếu cần thêm các thành phần mở rộng như `WhisperX`:

```powershell
pip install -r requirements-optional.txt
```

Nếu cần chạy test:

```powershell
pip install -r requirements-dev.txt
```

## Setup model local cho offline

Lần chạy đầu tiên nên dùng script bootstrap chuẩn:

```powershell
python scripts/setup_models.py --profile full
python scripts/check_installation.py --profile full
```

Hồ sơ `core` chỉ tải các model nền cho ASR/VAD/diarization:

```powershell
python scripts/setup_models.py --profile core
python scripts/check_installation.py --profile core
```

Toàn bộ model sau khi tải sẽ được lưu trong `models/` để dùng offline. Hướng dẫn chi tiết nằm ở [`docs/FIRST_RUN_OFFLINE_SETUP.md`](docs/FIRST_RUN_OFFLINE_SETUP.md).

## Chạy webapp

```powershell
python webapp.py
```

Mặc định webapp chạy tại `http://127.0.0.1:8000`.

Luồng khuyến nghị:

1. Chạy phiên âm cơ bản.
2. Nếu job dùng `whisper-v2`, có thể bật `Ổn định mốc thời gian`.
3. Nếu transcript còn méo/ngắt, bật các bước giảm lỗi theo nhu cầu.
4. Khi transcript đủ sạch, chạy `Tóm tắt trinh sát`.

## Chạy test nhẹ

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\test_webapp_api.py tests\test_stt_web_pipeline.py tests\test_stablets_adapter.py tests\test_whisperv2_adapter.py tests\test_audio_pipeline_fixes.py -q
```

## Benchmark local

Các script benchmark nhỏ đã có sẵn:

- `scripts/benchmark_stable_ts_small.py`
- `scripts/benchmark_stable_ts_longform_small.py`

Toàn bộ output benchmark và job runtime được ghi vào `output/` và đã được ignore khỏi Git.

## Ghi chú triển khai máy mới

- `models/`, `data/`, `output/`, môi trường ảo và cache đã được ignore.
- Webapp mặc định đi theo đường chạy `llama.cpp`, không yêu cầu `vllm`.
- Muốn chuyển sang máy không có internet, chỉ cần copy thêm cây thư mục `models/` sau khi đã chạy `scripts/setup_models.py` ở máy có mạng.

## Tài liệu liên quan

- [`docs/FIRST_RUN_OFFLINE_SETUP.md`](docs/FIRST_RUN_OFFLINE_SETUP.md)
- [`docs/OSS_STT_ASR_INTEGRATION_REVIEW_2026-04-10.md`](docs/OSS_STT_ASR_INTEGRATION_REVIEW_2026-04-10.md)
- [`docs/SPEECHTOINFOMATION_UPSTREAM_REVIEW_2026-04-11.md`](docs/SPEECHTOINFOMATION_UPSTREAM_REVIEW_2026-04-11.md)
- [`docs/UI-REVIEW.md`](docs/UI-REVIEW.md)
