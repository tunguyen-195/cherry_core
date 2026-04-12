# ĐỀ XUẤT NÂNG CẤP SOTA - CHERRY CORE V3
## Dựa trên Nghiên cứu Công nghệ Tiên tiến 2024-2025

**Ngày:** 2026-01-11
**Phản hồi lại:** Development Team Response

---

## 1. GHI NHẬN PHẢN HỒI CỦA DEVELOPMENT TEAM

### 1.1 Điểm đã được làm rõ

| Vấn đề | Đánh giá ban đầu | Phản hồi của Team | Kết luận |
|--------|------------------|-------------------|----------|
| Pyannote status | "Không chạy được" | **Đang chạy thành công** (53 segments) | Ghi nhận - đánh giá dựa trên output cũ |
| DER ước tính | ~35% | ~12% (với Pyannote 3.1) | Ghi nhận - cần benchmark chính thức |
| Offline capability | "Không thể" | **Có thể với cache strategy** | Ghi nhận - cần document rõ |
| VBx loop_prob | Đề xuất 0.75 | Giữ 0.45 (hội thoại nhanh) | Đồng ý - phù hợp Vietnamese conversational |

### 1.2 Điểm đồng thuận

- Vietnamese ASR errors cần domain-specific correction
- Architecture Clean và dễ mở rộng
- Repetition patterns cần cải thiện
- IntervalTree optimization hợp lý
- Test suite cần bổ sung

---

## 2. ĐÁNH GIÁ OVERFITTING VÀ GENERALIZATION

### 2.1 Rủi ro Overfitting hiện tại

**Test file hiện tại:** `test_audio.mp3` - Hotel booking conversation

| Đặc điểm | Giá trị | Rủi ro Generalization |
|----------|---------|----------------------|
| Số speakers | 2 | **Thấp** - Chỉ test với 2 speakers |
| Domain | Hotel booking | **Trung bình** - Vocabulary specific |
| Audio quality | Studio-quality | **Cao** - Không có noise thực tế |
| Accent | Chuẩn (Bắc/Nam) | **Trung bình** - Thiếu regional accents |
| Speaking style | Formal | **Trung bình** - Thiếu casual speech |
| Overlap | Minimal | **Cao** - Không test overlapping speech |

### 2.2 Các vấn đề Generalization cần kiểm tra

#### 2.2.1 Speaker Count Sensitivity

```
Benchmark cần thiết:
- 2 speakers (hiện tại) ✓
- 3-4 speakers (meeting scenario) ❌
- 5+ speakers (conference) ❌
- 1 speaker (monologue) ❌
```

#### 2.2.2 Audio Quality Robustness

```
Benchmark cần thiết:
- Studio quality (hiện tại) ✓
- Telephone quality (8kHz) ❌
- Far-field microphone ❌
- Background noise (SNR 10-20dB) ❌
- Echo/reverb ❌
```

#### 2.2.3 Speaking Style Diversity

```
Benchmark cần thiết:
- Formal conversation (hiện tại) ✓
- Casual conversation ❌
- Fast speech (>200 words/min) ❌
- Overlapping speech ❌
- Emotional speech ❌
```

### 2.3 Hardcoded Parameters - Rủi ro

**Các giá trị có thể gây overfitting:**

```python
# config.py - Cần đánh giá lại
class DiarizationConfig:
    SEGMENT_DURATION = 1.2   # Tối ưu cho 2-speaker hotel?
    STEP_DURATION = 0.2      # Tối ưu cho formal speech?
    MAX_SPEAKERS = 10        # OK

# silero_adapter.py
threshold = 0.3              # Tối ưu cho studio audio?
min_speech_duration_ms = 100 # Có thể miss short utterances?
min_silence_duration_ms = 300 # OK cho formal, có thể sai cho casual

# vbx_refiner.py
loop_prob = 0.45             # Đã được justify cho Vietnamese
```

### 2.4 Khuyến nghị Test Generalization

**Tạo benchmark suite với đa dạng audio:**

```yaml
# benchmark_config.yaml
test_sets:
  - name: "hotel_2speakers"
    audio: "samples/hotel_booking.mp3"
    speakers: 2
    quality: "studio"

  - name: "meeting_4speakers"
    audio: "samples/meeting_4p.mp3"
    speakers: 4
    quality: "meeting_room"

  - name: "phone_call"
    audio: "samples/phone_8khz.mp3"
    speakers: 2
    quality: "telephone"

  - name: "noisy_conversation"
    audio: "samples/cafe_noise.mp3"
    speakers: 2
    quality: "noisy"
    noise_snr: 15

  - name: "overlapping_speech"
    audio: "samples/debate.mp3"
    speakers: 3
    overlap_ratio: 0.2
```

---

## 3. CÔNG NGHỆ SOTA 2024-2025 - ĐỀ XUẤT NÂNG CẤP

### 3.1 WhisperX - End-to-End Pipeline (HIGHLY RECOMMENDED)

**Source:** [GitHub - m-bain/whisperX](https://github.com/m-bain/whisperX)

**Tại sao nên dùng:**
- Kết hợp **Whisper + Pyannote + wav2vec2** trong một pipeline thống nhất
- Word-level timestamps chính xác (wav2vec2 alignment)
- Đã xử lý integration giữa ASR và Diarization

**Architecture:**
```
Audio → faster-whisper → Coarse Segments
                ↓
        wav2vec2 Alignment → Word-level Timestamps
                ↓
        Pyannote 3.1 → Speaker Segments
                ↓
        Word-Speaker Assignment → Final Output
```

**Cài đặt:**
```bash
pip install whisperx
```

**Code Integration:**
```python
import whisperx

# 1. Transcribe with word timestamps
model = whisperx.load_model("large-v3", device, compute_type="float16")
result = whisperx.transcribe(model, audio_path, language="vi")

# 2. Align (wav2vec2)
align_model, metadata = whisperx.load_align_model(language_code="vi", device=device)
result = whisperx.align(result["segments"], align_model, metadata, audio_path, device)

# 3. Diarize (Pyannote)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
diarize_segments = diarize_model(audio_path)

# 4. Assign speakers to words
result = whisperx.assign_word_speakers(diarize_segments, result)
```

**Lợi ích:**
- Giảm complexity của codebase hiện tại
- Battle-tested với nhiều use cases
- Active community support

### 3.2 Pyannote 4.0 Community-1 (UPGRADE FROM 3.1)

**Source:** [Pyannote Community-1](https://www.pyannote.ai/blog/community-1)

**Cải tiến so với 3.1:**
- Giảm đáng kể **speaker confusion**
- Better **speaker counting** accuracy
- 40% faster training (if fine-tuning)
- Exclusive single-speaker mode eliminates overlap conflicts

**Migration:**
```python
# OLD (3.1)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# NEW (Community-1 / 4.0)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    use_auth_token=HF_TOKEN
)
```

**Benchmark Comparison:**

| Model | VoxConverse DER | AMI DER | Notes |
|-------|-----------------|---------|-------|
| speaker-diarization-3.1 | 11.2% | 18.8% | Current |
| **community-1** | **Better** | **Better** | Proposed |
| precision-2 (Premium) | Best | Best | Commercial |

### 3.3 Whisper large-v3-turbo (SPEED UPGRADE)

**Source:** [HuggingFace - whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

**Đặc điểm:**
- **6x faster** than large-v3
- **Same accuracy** as large-v2
- Decoder pruned: 32 → 4 layers
- Encoder unchanged (quality preserved)

**Benchmark (13 min audio):**

| Model | Time | WER |
|-------|------|-----|
| faster-whisper int8 | 52.6s | 4.594 |
| faster-distil-large-v3 | 22.5s | 2.392 |
| **faster-large-v3-turbo** | **19.1s** | **1.919** |

**Implementation:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio_path, language="vi")
```

### 3.4 NeMo Sortformer (ALTERNATIVE END-TO-END)

**Source:** [NVIDIA NeMo Speaker Diarization](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html)

**Đặc điểm:**
- End-to-end Transformer-based diarization
- Không cần clustering riêng
- Overlap-aware natively

**Khi nào dùng:**
- Có GPU NVIDIA mạnh (A100/H100)
- Cần xử lý overlapping speech nhiều
- Muốn fine-tune trên Vietnamese data

### 3.5 Vietnamese ASR - PhoWhisper (KEEP CURRENT)

**Source:** [VinAI PhoWhisper](https://github.com/VinAIResearch/PhoWhisper)

**Đánh giá:** PhoWhisper vẫn là SOTA cho Vietnamese ASR.

**Khuyến nghị:**
- Giữ PhoWhisper-large cho accuracy
- Hoặc dùng Whisper large-v3-turbo + Vietnamese post-processing

### 3.6 Vietnamese Spell Correction - VSEC & Hierarchical Transformer

**Sources:**
- [VSEC Paper](https://arxiv.org/pdf/2111.00640)
- [VinAI Grammar](https://grammar.vinai.io/)

**SOTA Models:**

| Model | Approach | Availability |
|-------|----------|--------------|
| **Hierarchical Transformer** | Char + Word level | [grammar.vinai.io](https://grammar.vinai.io/) |
| VSEC | Seq2Seq + BPE | [Paper](https://arxiv.org/pdf/2111.00640) |
| BARTpho | Seq2Seq pretrained | [HuggingFace](https://huggingface.co/vinai/bartpho-syllable) |

**Đề xuất:** Thay ProtonX bằng VinAI Grammar API hoặc fine-tune BARTpho.

---

## 4. ĐỀ XUẤT KIẾN TRÚC MỚI - CHERRY CORE V3

### 4.1 Option A: WhisperX Integration (RECOMMENDED)

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHERRY CORE V3 - WhisperX                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Audio     │                                                │
│  │   Input     │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     WhisperX Pipeline                        ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  ││
│  │  │ faster-whisper│→│ wav2vec2      │→│ Pyannote 4.0    │  ││
│  │  │ large-v3-turbo│  │ Alignment     │  │ Community-1     │  ││
│  │  │ (Vietnamese)  │  │ (Word-level)  │  │ (Speaker ID)    │  ││
│  │  └───────────────┘  └───────────────┘  └─────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Vietnamese Post-Processing                      ││
│  │  1. Domain-specific corrections (Hotel, Legal, Medical)     ││
│  │  2. VinAI Grammar API / BARTpho                             ││
│  │  3. PII formatting                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Strategic Analysis                        ││
│  │  Vistral 7B / Qwen3 8B (Offline LLM)                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Option B: Keep Current Architecture (Enhanced)

```
┌─────────────────────────────────────────────────────────────────┐
│                 CHERRY CORE V3 - Enhanced Current                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │   Audio     │───▶│  Silero VAD │                            │
│  │   Input     │    │  (Offline)  │                            │
│  └─────────────┘    └──────┬──────┘                            │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ ASR         │    │ Diarization │    │ Word Align  │        │
│  │ (faster-    │    │ (Pyannote   │    │ (stable-ts) │        │
│  │ whisper-    │    │ Community-1)│    │             │        │
│  │ turbo)      │    │             │    │             │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Word-Speaker Alignment (IntervalTree)           ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Vietnamese Post-Processing                      ││
│  │  - Domain corrections                                        ││
│  │  - BARTpho / VinAI Grammar                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. IMPLEMENTATION PRIORITIES

### Priority 1: Generalization Testing (CRITICAL)

```python
# tests/test_generalization.py

import pytest
from pathlib import Path

class TestGeneralization:
    """Test system on diverse audio conditions."""

    BENCHMARK_AUDIOS = [
        ("hotel_2speakers", 2, "studio"),
        ("meeting_4speakers", 4, "meeting"),
        ("phone_call", 2, "telephone"),
        ("noisy_cafe", 2, "noisy"),
        ("overlapping_debate", 3, "overlap"),
    ]

    @pytest.mark.parametrize("name,expected_speakers,condition", BENCHMARK_AUDIOS)
    def test_speaker_count(self, name, expected_speakers, condition):
        """Test speaker count accuracy across conditions."""
        audio_path = Path(f"benchmark/{name}.wav")
        result = diarize(audio_path)

        detected_speakers = len(set(s.speaker_id for s in result))

        # Allow +/- 1 speaker tolerance
        assert abs(detected_speakers - expected_speakers) <= 1

    def test_der_across_conditions(self):
        """Ensure DER doesn't spike on specific conditions."""
        results = {}
        for name, _, _ in self.BENCHMARK_AUDIOS:
            der = calculate_der(f"benchmark/{name}.wav")
            results[name] = der

        # No single condition should have DER > 30%
        for name, der in results.items():
            assert der < 0.30, f"DER too high for {name}: {der}"

        # Standard deviation should be low (consistent performance)
        import numpy as np
        std_dev = np.std(list(results.values()))
        assert std_dev < 0.10, f"Inconsistent DER across conditions: std={std_dev}"
```

### Priority 2: Upgrade to Pyannote Community-1

```python
# infrastructure/adapters/diarization/pyannote_v4_adapter.py

class PyannoteV4Adapter(ISpeakerDiarizer):
    """
    Upgraded to Pyannote 4.0 Community-1.
    Better speaker counting and assignment than 3.1.
    """

    def __init__(self, hf_token: str = None, **kwargs):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            # Use Community-1 (4.0) instead of 3.1
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=self.hf_token
            )

            # Optional: Enable exclusive single-speaker mode
            # self._pipeline.instantiate({"segmentation": {"exclusive": True}})
```

### Priority 3: WhisperX Integration (Optional)

```python
# infrastructure/adapters/asr/whisperx_adapter.py

import whisperx

class WhisperXAdapter(ITranscriber, ISpeakerDiarizer):
    """
    End-to-end ASR + Diarization using WhisperX.
    Replaces separate ASR and Diarization adapters.
    """

    def __init__(self,
                 model_size: str = "large-v3-turbo",
                 hf_token: str = None,
                 device: str = "cuda"):
        self.model_size = model_size
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = device

    def transcribe_and_diarize(self, audio_path: str) -> List[SpeakerSegment]:
        """Full pipeline in one call."""

        # 1. Load and transcribe
        model = whisperx.load_model(self.model_size, self.device, compute_type="float16")
        result = whisperx.transcribe(model, audio_path, language="vi")

        # 2. Align words (wav2vec2)
        align_model, metadata = whisperx.load_align_model(
            language_code="vi", device=self.device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio_path, self.device
        )

        # 3. Diarize
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        diarize_segments = diarize_model(audio_path)

        # 4. Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 5. Convert to SpeakerSegment
        return self._convert_to_segments(result)
```

### Priority 4: Vietnamese Post-Processing Enhancement

```python
# infrastructure/adapters/correction/vinai_corrector.py

import requests

class VinAIGrammarAdapter(ITextCorrector):
    """
    Use VinAI Grammar API for Vietnamese correction.
    https://grammar.vinai.io/
    """

    API_URL = "https://grammar.vinai.io/api/correct"

    def correct(self, text: str) -> str:
        try:
            response = requests.post(
                self.API_URL,
                json={"text": text},
                timeout=10
            )
            if response.ok:
                return response.json().get("corrected_text", text)
        except Exception as e:
            logger.warning(f"VinAI Grammar API failed: {e}")

        return text


class BARTphoCorrector(ITextCorrector):
    """
    Offline Vietnamese correction using BARTpho.
    """

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")

    def correct(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 6. SUMMARY - ACTION ITEMS

### Immediate (Sprint 1)

| Task | Priority | Effort |
|------|----------|--------|
| Create benchmark suite với đa dạng audio | CRITICAL | 2-3 days |
| Run generalization tests | CRITICAL | 1 day |
| Document offline cache strategy for Pyannote | HIGH | 0.5 day |

### Short-term (Sprint 2)

| Task | Priority | Effort |
|------|----------|--------|
| Upgrade Pyannote 3.1 → Community-1 | HIGH | 1 day |
| Integrate faster-whisper large-v3-turbo | HIGH | 1 day |
| Add IntervalTree to AlignmentService | MEDIUM | 0.5 day |

### Medium-term (Sprint 3-4)

| Task | Priority | Effort |
|------|----------|--------|
| Evaluate WhisperX integration | MEDIUM | 2 days |
| Replace ProtonX with BARTpho/VinAI Grammar | MEDIUM | 2 days |
| Complete test suite | MEDIUM | 2-3 days |

### Long-term (Future)

| Task | Priority | Effort |
|------|----------|--------|
| Fine-tune Pyannote on Vietnamese data | LOW | 1-2 weeks |
| Evaluate NeMo Sortformer | LOW | 3-5 days |
| Production optimization (batching, GPU) | LOW | 1 week |

---

## 7. REFERENCES

### Academic Papers
- [Pyannote 3.1 Paper - ICASSP 2023](https://arxiv.org/abs/2212.02060)
- [PhoWhisper - ICLR 2024](https://arxiv.org/pdf/2406.02555)
- [VSEC - Vietnamese Spelling Correction](https://arxiv.org/pdf/2111.00640)

### Official Documentation
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Pyannote Community-1 Blog](https://www.pyannote.ai/blog/community-1)
- [NVIDIA NeMo Speaker Diarization](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html)
- [VinAI PhoWhisper](https://github.com/VinAIResearch/PhoWhisper)

### Benchmarks & Comparisons
- [Speaker Diarization Benchmark - Picovoice](https://github.com/Picovoice/speaker-diarization-benchmark)
- [Pyannote Evaluation Guide](https://www.pyannote.ai/blog/how-to-evaluate-speaker-diarization-performance)
- [Modal - Open Source STT Comparison 2025](https://modal.com/blog/open-source-stt)

---

*Báo cáo được tạo dựa trên nghiên cứu công nghệ SOTA 2024-2025*
*Ngày: 2026-01-11*
