# 🍒 Cherry Core V2 - Project Context Summary

> **Cập nhật**: 2026-01-27 | **Version**: V2 | **Status**: Development

---

## 🎯 Mục Tiêu Dự Án

**Forensic Audio Intelligence System** - Hệ thống phân tích audio forensic cho tiếng Việt:

1. **Transcription**: Chuyển đổi giọng nói → văn bản (WER ~4.67%)
2. **Speaker Diarization**: Nhận diện người nói
3. **Text Correction**: Sửa lỗi chính tả tiếng Việt  
4. **LLM Analysis**: Phân tích forensic với AI
5. **Report Generation**: Tạo báo cáo JSON có cấu trúc

---

## 🏗️ Kiến Trúc

**Pattern**: Hexagonal Architecture (Ports & Adapters)

```
┌──────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                              │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐    │
│  │ Transcript  │ │SpeakerSegment│ │  StrategicReport    │    │
│  └─────────────┘ └─────────────┘ └──────────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│                      PORTS (Interfaces)                       │
│  ITranscriber  │ ILLMEngine │ ISpeakerDiarizer │ ITextCorrector│
├──────────────────────────────────────────────────────────────┤
│                     ADAPTERS (Implementations)                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │ PhoWhisper  │ │ LlamaCpp    │ │ Pyannote    │             │
│  │ WhisperV3   │ │ vLLM        │ │ Resemblyzer │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Components

### Domain Entities

| Entity | Mô tả |
|--------|-------|
| `Transcript` | Kết quả ASR với text + segments + metadata |
| `TranscriptSegment` | Segment với start/end/text |
| `SpeakerSegment` | Segment với speaker_id, word-level timestamps |
| `StrategicReport` | Báo cáo forensic với tactical/behavioral intelligence |

### Port Interfaces

| Port | Method | Mô tả |
|------|--------|-------|
| `ITranscriber` | `transcribe(audio_path)` | Audio → Transcript |
| `ILLMEngine` | `generate(prompt)`, `load()` | Text generation |
| `ISpeakerDiarizer` | `diarize(audio_path)` | Audio → SpeakerSegments |
| `ITextCorrector` | `correct(text)` | Vietnamese spelling fix |

### Adapters

| Adapter | Type | Status |
|---------|------|--------|
| `phowhisper_adapter.py` | ASR | ✅ Working |
| `whisperv3_adapter.py` | ASR | ✅ Working (safetensors) |
| `hallucination_filter.py` | ASR | ✅ Working |
| `llamacpp_adapter.py` | LLM | ✅ Working (Windows) |
| `vllm_adapter.py` | LLM | ✅ Working (WSL2) |

---

## 🧪 Test Scripts

| Script | Purpose |
|--------|---------|
| `analyze_manual_transcript.py` | Canary test LLM |
| `full_pipeline.py` | Full ASR+LLM |
| `step1_transcribe.py` | Transcription only |
| `step2_diarize.py` | Diarization only |
| `test_user_audio.py` | Real audio test |

---

## 📊 Optimization Cycles

| Cycle | Focus | Status |
|-------|-------|--------|
| 1. GBNF Grammar | ✅ Complete |
| 2. Speaker Diarization | ⏳ In Progress |
| 3. Chain-of-Verification | ⏸️ Pending |
| 4. vLLM Serving | ✅ Complete |
