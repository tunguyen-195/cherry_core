# BÁO CÁO ĐÁNH GIÁ CHI TIẾT - CHERRY CORE V2
## Hệ thống Forensic AI cho Vietnamese Speech-to-Text & Speaker Diarization

**Ngày đánh giá:** 2026-01-11
**Phiên bản:** Cherry Core V2
**Đánh giá viên:** AI Research Assistant

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Đánh giá kết quả Diarization](#2-đánh-giá-kết-quả-diarization)
3. [Đánh giá ASR (Speech-to-Text)](#3-đánh-giá-asr-speech-to-text)
4. [Đánh giá khả năng hoạt động Offline](#4-đánh-giá-khả-năng-hoạt-động-offline)
5. [Phân tích kỹ thuật chuyên sâu](#5-phân-tích-kỹ-thuật-chuyên-sâu)
6. [Đề xuất nâng cấp](#6-đề-xuất-nâng-cấp)
7. [Kết luận](#7-kết-luận)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Kiến trúc hệ thống

Cherry Core V2 được thiết kế theo **Clean Architecture** với các layer rõ ràng:

```
cherry_core/
├── core/                    # Domain Layer (Entities, Ports)
│   ├── domain/entities.py   # Transcript, SpeakerSegment, StrategicReport
│   ├── ports/               # Interface definitions (ITranscriber, ISpeakerDiarizer, etc.)
│   └── config.py            # System configuration
├── infrastructure/          # Implementation Layer
│   └── adapters/
│       ├── asr/             # WhisperV2, WhisperV3, PhoWhisper
│       ├── diarization/     # Pyannote, SpeechBrain, Resemblyzer, VBx
│       ├── vad/             # Silero VAD
│       ├── llm/             # LlamaCpp, vLLM
│       └── correction/      # ProtonX Vietnamese Spell Correction
├── application/             # Use Cases
│   └── use_cases/
│       ├── transcribe_audio.py
│       └── generate_report.py
└── presentation/            # CLI Interface
```

### 1.2 Các thành phần chính

| Component | Mục đích | Model/Thư viện |
|-----------|----------|----------------|
| ASR | Speech-to-Text | Whisper V2/V3, PhoWhisper |
| VAD | Voice Activity Detection | Silero VAD |
| Diarization | Speaker Identification | Pyannote 3.1, SpeechBrain ECAPA-TDNN |
| LLM | Strategic Analysis | Vistral 7B, Qwen3 8B |
| Correction | Spell Correction | ProtonX Legal-TC |

---

## 2. ĐÁNH GIÁ KẾT QUẢ DIARIZATION

### 2.1 So sánh kết quả

**File kết quả thực tế:** `test_audio_diarized_final.txt`
**File kết quả đúng (Ground Truth):** `sample/Result.txt`

#### 2.1.1 Phân tích chi tiết

| Tiêu chí | Ground Truth | Kết quả thực tế | Đánh giá |
|----------|--------------|-----------------|----------|
| **Số lượng speaker** | 2 (Speaker 0, Speaker 1) | 3 (Receptionist, Customer, Unknown) | ⚠️ Sai - phát hiện thêm "Unknown" |
| **Độ phân giải thời gian** | ~1-5 giây/segment | 5-15 giây/segment | ❌ Kém - segments quá dài |
| **Speaker assignment** | Chính xác theo từng câu | Nhiều câu gộp chung sai speaker | ❌ Kém |
| **Timestamp accuracy** | ms precision | 5-15s resolution | ❌ Kém hơn nhiều |

#### 2.1.2 Các lỗi cụ thể được phát hiện

**Lỗi 1: Segment đầu tiên gộp quá nhiều nội dung**
```
Ground Truth:
00:00:00,100 --> 00:00:20,419 [Speaker 0] Khách sạn JW Mariott...
00:00:20,420 --> 00:00:24,900 [Speaker 1] Chào em nhé, chị à...

Kết quả thực tế:
00:00:00,000 --> 00:00:15,000 [Unknown] Xin ký chào quý khách... Chào em nhé. Chị muốn đặt phòng...
```
→ **Vấn đề:** Gộp cả lời của Receptionist VÀ Customer vào 1 segment

**Lỗi 2: Speaker assignment không nhất quán**
- Ground Truth phân biệt rõ: Speaker 0 = Lễ tân, Speaker 1 = Khách hàng
- Kết quả thực tế: Nhiều đoạn bị đánh nhãn "Unknown" hoặc hoán đổi giữa Receptionist/Customer

**Lỗi 3: Mất thông tin cuối cuộc gọi**
- Ground Truth: 101 dòng, bao gồm "Vi xuống. OK!" (nội bộ)
- Kết quả thực tế: 105 dòng nhưng thiếu một số chi tiết

### 2.2 Nguyên nhân gốc rễ

1. **SpeechBrain sliding window quá dài (1.2s segment, 0.2s step)**
   - Dẫn đến segments thô, không bắt được speaker change nhanh

2. **Clustering algorithm không tối ưu**
   - Spectral/Agglomerative clustering với Eigengap heuristic không hoạt động tốt cho tiếng Việt

3. **VBx Refiner loop_prob=0.45 quá cao**
   - Over-smoothing, bỏ qua các speaker change ngắn

4. **Pyannote không được sử dụng (thiếu HF_TOKEN)**
   - Config mặc định ENGINE="pyannote" nhưng fallback về SpeechBrain

### 2.3 Điểm đánh giá Diarization

| Metric | Score | Thang điểm |
|--------|-------|------------|
| Speaker Count Accuracy | 2/3 | **67%** |
| Speaker Assignment | ~40% | **Kém** |
| Temporal Resolution | ~20% | **Rất kém** |
| Overall DER (ước tính) | >35% | **Không đạt SOTA** |

**SOTA Reference:** Pyannote 3.1 đạt ~11% DER trên benchmark chuẩn

---

## 3. ĐÁNH GIÁ ASR (SPEECH-TO-TEXT)

### 3.1 So sánh Transcription

**Ground Truth:**
```
Khách sạn JW Mariott Hotel Hà Nội xin kính chào quý khách!
```

**Kết quả thực tế:**
```
Xin ký chào quý khách. Tôi là Tâm, nhân viên cộng phận lễ tân.
```

### 3.2 Các lỗi ASR điển hình

| Loại lỗi | Ví dụ | Ground Truth |
|----------|-------|--------------|
| **Homophone** | "điều trú" | "lưu trú" |
| **Vietnamese tone** | "Xin ký" | "xin kính" |
| **Named entity** | "G.W. Marriott" | "JW Mariott" |
| **Số/Digit** | "09121212" | Thiếu số |
| **Domain-specific** | "đi lặn", "vòng x kế tiếp" | "Deluxe", "Executive" |
| **Repetition hallucination** | "Quyên. Quyên. Quyên..." | "Quyên" (1 lần) |

### 3.3 Đánh giá các model ASR

| Model | Ưu điểm | Nhược điểm | Điểm |
|-------|---------|------------|------|
| **Whisper V2** | Anti-hallucination, VAD | Lỗi tiếng Việt nhiều | 6/10 |
| **Whisper V3** | Tốt hơn V2 một chút | Cần local_files_only | 6.5/10 |
| **PhoWhisper** | Chuyên tiếng Việt | Cần HuggingFace download | 7.5/10 |

### 3.4 Anti-Hallucination Features

Dự án đã implement:
- ✅ `condition_on_previous_text=False`
- ✅ `compression_ratio_threshold=2.0`
- ✅ `no_speech_threshold=0.5`
- ✅ Repetition pattern removal (regex)
- ✅ Silero VAD preprocessing

**Nhưng vẫn còn hallucination:** `"Quyên. Quyên. Quyên..."` vẫn xuất hiện trong output

---

## 4. ĐÁNH GIÁ KHẢ NĂNG HOẠT ĐỘNG OFFLINE

### 4.1 Kiểm tra các thành phần

| Component | Offline Ready? | Vấn đề |
|-----------|----------------|--------|
| **Whisper V2** | ✅ Có | Cần file `large-v2.pt` |
| **Whisper V3** | ✅ Có | Cần `local_files_only=True` |
| **PhoWhisper** | ⚠️ Có điều kiện | Fallback sang HuggingFace nếu thiếu local |
| **Pyannote 3.1** | ❌ KHÔNG | **Cần HF_TOKEN và Internet** |
| **SpeechBrain** | ✅ Có | Cần download trước ECAPA-TDNN |
| **Silero VAD** | ✅ Có | Cần `silero_vad.jit` + `utils_vad.py` |
| **LlamaCpp (Vistral)** | ✅ Có | Cần `.gguf` file |
| **ProtonX** | ⚠️ Có điều kiện | Cần download model trước |

### 4.2 Các vấn đề Offline quan trọng

#### 4.2.1 Pyannote (CRITICAL)
```python
# pyannote_adapter.py:57
self._pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=self.hf_token  # ❌ Cần token + Internet
)
```
**Giải pháp:** Không có cách nào chạy Pyannote offline hoàn toàn do model được gated trên HuggingFace

#### 4.2.2 PhoWhisper fallback
```python
# phowhisper_adapter.py:26
model_id = str(PHOWHISPER_PATH) if PHOWHISPER_PATH.exists() else "vinai/phowhisper-large"
# ❌ Nếu local không tồn tại, sẽ download từ HF
```

### 4.3 Đánh giá tổng thể Offline

| Trạng thái | Chi tiết |
|------------|----------|
| **Với Pyannote** | ❌ Không hoạt động offline |
| **Với SpeechBrain** | ✅ Có thể offline (nhưng quality kém hơn) |
| **ASR + LLM** | ✅ Hoạt động offline nếu download model trước |

**Khuyến nghị:** Nếu cần 100% offline, phải sử dụng SpeechBrain thay vì Pyannote

---

## 5. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU

### 5.1 Speaker Diarization Pipeline

```
Audio → Silero VAD → Speech Regions → ECAPA-TDNN Embeddings → Spectral Clustering → VBx Refinement → Labels
```

#### 5.1.1 ECAPA-TDNN (SpeechBrain)
- **Architecture:** Emphasized Channel Attention, Propagation and Aggregation
- **Embedding dimension:** 192-d speaker vectors
- **Training data:** VoxCeleb1, VoxCeleb2 (English-centric)

**Vấn đề với tiếng Việt:**
- Model được train chủ yếu trên tiếng Anh
- Vietnamese có nhiều tone, pitch variation → confuse embedding

#### 5.1.2 Clustering Analysis

**Spectral Clustering với Cosine Affinity:**
```python
SpectralClustering(n_clusters=k, affinity="cosine", assign_labels="discretize")
```

**Eigengap Heuristic cho K estimation:**
- Tính affinity matrix
- Compute eigenvalues của Laplacian
- K = argmax(gap between consecutive eigenvalues)

**Vấn đề:** Eigengap không ổn định với short segments

#### 5.1.3 VBx Resegmentation

```python
VBxRefiner(loop_prob=0.45)  # Viterbi-based smoothing
```

**Vấn đề với loop_prob=0.45:**
- Xác suất chuyển speaker = (1-0.45)/(N-1) = 27.5% (với N=2)
- Quá cao, dẫn đến over-smoothing

### 5.2 ASR Pipeline Analysis

```
Audio → (Optional) Silero VAD → Whisper Decoding → Repetition Filter → Transcript
```

#### 5.2.1 Whisper Decoding Parameters
```python
result = self.model.transcribe(
    processed_audio,
    language="vi",
    beam_size=5,
    best_of=5,
    temperature=0.0,  # Deterministic
    condition_on_previous_text=False,  # Anti-hallucination
    compression_ratio_threshold=2.0,
    logprob_threshold=-1.0,
    no_speech_threshold=0.5,
)
```

#### 5.2.2 Repetition Pattern
```python
REPETITION_PATTERN = re.compile(r'(\b\w+\b)(\s*[.,]?\s*\1){2,}', re.IGNORECASE)
```
**Vấn đề:** Pattern quá đơn giản, không bắt được các repetition phức tạp

### 5.3 LLM Analysis Pipeline

```python
# Vistral 7B với GBNF Grammar
llm(prompt, max_tokens=4096, temperature=0.1, grammar=grammar)
```

**Ưu điểm:**
- Vietnamese-native LLM (Vistral)
- GBNF grammar constraint cho structured output
- Offline capable với GGUF format

---

## 6. ĐỀ XUẤT NÂNG CẤP

### 6.1 Cải thiện Diarization (CRITICAL)

#### 6.1.1 Sử dụng Pyannote 3.1 (Recommended SOTA)
```python
# Yêu cầu:
# 1. HuggingFace token
# 2. Accept license tại: https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Set HF_TOKEN environment variable

export HF_TOKEN="your_token_here"
```

**Lợi ích:**
- 16ms frame resolution (vs 1.2s của SpeechBrain)
- Neural speaker change detection
- ~11% DER (vs ~35% hiện tại)

#### 6.1.2 Cải thiện SpeechBrain (Nếu cần offline)

```python
# config.py - Điều chỉnh parameters
class DiarizationConfig:
    SEGMENT_DURATION = 0.5   # Giảm từ 1.2s xuống 0.5s
    STEP_DURATION = 0.1      # Giảm từ 0.2s xuống 0.1s
    CLUSTERING_TYPE = "agglomerative"  # Thử Agglomerative thay vì Spectral

# vbx_refiner.py - Giảm loop_prob
VBxRefiner(loop_prob=0.7)  # Tăng từ 0.45 lên 0.7 để ít switching hơn
```

#### 6.1.3 Word-level Alignment
```python
# Sử dụng whisper-timestamped hoặc stable-ts
import whisper_timestamped as whisper

result = whisper.transcribe(model, audio_path, language="vi")
# result["words"] chứa word-level timestamps
```

### 6.2 Cải thiện ASR

#### 6.2.1 Sử dụng faster-whisper
```python
# Thay thế openai-whisper bằng faster-whisper
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio_path, language="vi", vad_filter=True)
```

**Lợi ích:**
- 4x faster inference
- Built-in VAD filter
- Word-level timestamps

#### 6.2.2 Cải thiện Repetition Detection
```python
import re

def advanced_repetition_filter(text: str) -> str:
    """
    Multi-pattern repetition detection for Vietnamese.
    """
    patterns = [
        # Word repetition: "Quyên. Quyên. Quyên."
        r'(\b[\w\u00C0-\u1EF9]+\b)(\s*[.,!?]?\s*\1){2,}',
        # Phrase repetition: "xin chào. xin chào."
        r'([\w\u00C0-\u1EF9\s]{3,20})(\s*[.,!?]?\s*\1){1,}',
        # Character repetition: "aaaaaa"
        r'(.)\1{4,}',
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, r'\1', result, flags=re.IGNORECASE)

    return re.sub(r'\s+', ' ', result).strip()
```

#### 6.2.3 Vietnamese-specific Post-processing
```python
# Dictionary-based correction cho domain-specific terms
HOTEL_CORRECTIONS = {
    "đi lặn": "Deluxe",
    "vòng x kế tiếp": "Executive",
    "x kế tiếp": "Executive",
    "điều trú": "lưu trú",
    "cộng phận": "bộ phận",
    "căn cứ công dân": "căn cước công dân",
    "quỷ trả": "hủy trả",
}
```

### 6.3 Cải thiện Offline Capability

#### 6.3.1 Pre-download Script
```python
# scripts/setup_models.py
def download_all_models():
    """Download tất cả models cần thiết cho offline operation."""

    # 1. Whisper V2
    import whisper
    whisper.load_model("large-v2", download_root="models/")

    # 2. PhoWhisper
    from transformers import AutoModelForSpeechSeq2Seq
    AutoModelForSpeechSeq2Seq.from_pretrained(
        "vinai/phowhisper-large",
        cache_dir="models/phowhisper"
    )

    # 3. SpeechBrain ECAPA-TDNN
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/speechbrain"
    )

    # 4. Silero VAD
    torch.hub.download_url_to_file(
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.jit",
        "models/silero/silero_vad.jit"
    )
```

### 6.4 Đề xuất Architecture mới

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHERRY CORE V3 (Proposed)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Audio     │───▶│  Silero VAD │───▶│ faster-whisper      │ │
│  │   Input     │    │  (Offline)  │    │ (Word timestamps)   │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    WORD-LEVEL ALIGNMENT                      ││
│  │  words: [{word: "Xin", start: 0.0, end: 0.3}, ...]          ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  SPEAKER DIARIZATION                         ││
│  │  ┌───────────────┐  OR  ┌─────────────────────────────────┐ ││
│  │  │ Pyannote 3.1  │      │ SpeechBrain + VBx (Offline)     │ ││
│  │  │ (16ms, SOTA)  │      │ (0.5s segments, tuned params)   │ ││
│  │  └───────────────┘      └─────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              WORD-TO-SPEAKER ALIGNMENT                       ││
│  │  For each word: find overlapping speaker segment             ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 POST-PROCESSING PIPELINE                     ││
│  │  1. Vietnamese spell correction (ProtonX)                    ││
│  │  2. Domain-specific term replacement                         ││
│  │  3. Repetition filter (advanced patterns)                    ││
│  │  4. PII detection & formatting                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    FINAL OUTPUT                              ││
│  │  - Diarized transcript with accurate timestamps              ││
│  │  - Speaker-labeled turns                                     ││
│  │  - Strategic analysis (LLM)                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. KẾT LUẬN

### 7.1 Tổng kết đánh giá

| Thành phần | Điểm | Đánh giá |
|------------|------|----------|
| **Architecture** | 8/10 | Clean Architecture, dễ mở rộng |
| **ASR Quality** | 6/10 | Còn nhiều lỗi tiếng Việt |
| **Diarization** | 4/10 | DER cao, timestamp thô |
| **Offline Capability** | 6/10 | Pyannote cần internet |
| **Code Quality** | 7/10 | Well-structured, có logging |

### 7.2 Ưu tiên cải thiện

1. **[CRITICAL]** Sử dụng Pyannote 3.1 hoặc tune SpeechBrain parameters
2. **[HIGH]** Implement word-level alignment với stable-ts/whisper-timestamped
3. **[HIGH]** Chuyển sang faster-whisper cho performance
4. **[MEDIUM]** Cải thiện repetition detection patterns
5. **[MEDIUM]** Thêm domain-specific Vietnamese corrections
6. **[LOW]** Refactor requirements.txt với version pinning

### 7.3 Kết luận cuối cùng

Cherry Core V2 có **kiến trúc tốt** và **concept đúng hướng**, nhưng kết quả thực tế **chưa đạt production quality**:

- **Diarization:** Cần chuyển sang Pyannote 3.1 hoặc tune lại SpeechBrain nghiêm túc
- **ASR:** Whisper hoạt động nhưng cần post-processing mạnh hơn cho tiếng Việt
- **Offline:** Khó đạt được với Pyannote, cần trade-off quality vs offline

**Khuyến nghị:** Đầu tư thời gian vào việc tune Diarization trước, vì đây là bottleneck chính ảnh hưởng đến chất lượng output cuối cùng.

---

## PHỤ LỤC

### A. Files được đánh giá

```
cherry_core/
├── core/config.py
├── core/domain/entities.py
├── core/ports/*.py
├── infrastructure/adapters/asr/*.py
├── infrastructure/adapters/diarization/*.py
├── infrastructure/adapters/vad/silero_adapter.py
├── infrastructure/adapters/llm/*.py
├── infrastructure/adapters/correction/*.py
├── infrastructure/factories/system_factory.py
├── application/use_cases/*.py
├── presentation/cli/main.py
├── output/test_audio_diarized_final.txt
└── output/sample/Result.txt
```

### B. Metrics References

- **DER (Diarization Error Rate):** Chuẩn đánh giá speaker diarization
  - SOTA (Pyannote 3.1): ~11%
  - Acceptable: <20%
  - Poor: >30%

- **WER (Word Error Rate):** Chuẩn đánh giá ASR
  - Whisper large-v3 Vietnamese: ~15-20%
  - PhoWhisper: ~10-15%

### C. Thư viện đề xuất bổ sung

```txt
# requirements.txt (proposed additions)
faster-whisper>=1.0.0
whisper-timestamped>=1.14.0
pyannote.audio>=3.1.0
stable-ts>=2.0.0
speechbrain>=1.0.0
torch>=2.0.0
transformers>=4.36.0
```

---

*Báo cáo được tạo tự động bởi AI Research Assistant*
