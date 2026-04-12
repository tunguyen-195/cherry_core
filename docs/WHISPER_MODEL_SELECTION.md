# WHISPER MODEL SELECTION GUIDE

## Why Whisper V2 is Preferred over V3 for Cherry Core

**Ngày:** 2026-01-11
**Dựa trên:** Research và thực nghiệm

---

## 1. TÓM TẮT

**Kết luận:** Whisper Large-V2 được chọn làm default ASR engine cho Cherry Core vì:

- V3 có vấn đề hallucination nghiêm trọng (4x nhiều hơn V2)
- V2 ổn định hơn cho tiếng Việt
- Anti-hallucination settings hoạt động tốt hơn với V2

---

## 2. BẰNG CHỨNG RESEARCH

### 2.1 Deepgram Study (2024)

| Metric | Whisper V2 | Whisper V3 |
|--------|------------|------------|
| Hallucination frequency | 1x (baseline) | **4x worse** |
| Median WER (real-world) | 12.7 | 53.4 |
| Repetition issues | Có | **Tệ hơn** |

> "Whisper Large-v3 hallucinates **four times more frequently** than v2"
> — Deepgram Research

### 2.2 Các loại Hallucination V3

1. **Repetition**: "Quyên. Quyên. Quyên..." (lặp vô hạn)
2. **Fabrication**: Thêm nội dung không có trong audio
3. **YouTube bias**: "Thank you for watching" trong silent segments
4. **Named entities**: Thêm tên người không tồn tại

### 2.3 Nguyên nhân

1. **Training data noisy**: V3 train trên dữ liệu chất lượng kém hơn
2. **Seq2seq prediction**: Model predict next word thay vì transcribe
3. **Silent segment sensitivity**: Hallucinate khi gặp silence
4. **Low-resource language**: Tiếng Việt ít training data

---

## 3. KINH NGHIỆM THỰC TẾ CHERRY CORE

### 3.1 Test với test_audio.mp3

| Issue | V3 Result | V2 Result |
|-------|-----------|-----------|
| Repetition "Quyên..." | ❌ Có | ✅ Không |
| "Thank you for watching" | ❌ Có | ✅ Không |
| Overall quality | Kém | Tốt |

### 3.2 Configuration hoạt động tốt với V2

```python
# whisperv2_adapter.py
result = self.model.transcribe(
    audio,
    language="vi",
    beam_size=5,
    best_of=5,
    temperature=0.0,  # Deterministic
    condition_on_previous_text=False,  # Anti-hallucination
    compression_ratio_threshold=2.0,
    no_speech_threshold=0.5,
)
```

---

## 4. KHI NÀO DÙNG V3?

### 4.1 Có thể dùng V3 nếu

- Có VAD preprocessing mạnh (Silero)
- Audio quality cao (studio)
- Không có silent segments dài
- Sử dụng faster-whisper với vad_filter=True

### 4.2 Whisper V3-Turbo

V3-Turbo có thể là lựa chọn tốt hơn V3 vì:

- 6x faster
- Decoder pruned (32 → 4 layers)
- Ít hallucination hơn V3 full

```python
# Nếu muốn thử V3-Turbo
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cuda")
segments, _ = model.transcribe(audio, language="vi", vad_filter=True)
```

---

## 5. BEST PRACTICES

### 5.1 Luôn dùng VAD preprocessing

```python
# Silero VAD trước khi transcribe
vad_adapter = SileroVADAdapter()
speech_timestamps = vad_adapter.get_speech_timestamps(audio_path)
```

### 5.2 Anti-hallucination settings

```python
condition_on_previous_text=False  # QUAN TRỌNG
compression_ratio_threshold=2.0
no_speech_threshold=0.5
```

### 5.3 Repetition filter

```python
REPETITION_PATTERN = re.compile(r'(\b[\w\u00C0-\u1EF9]+\b)(\s*[.,!?]?\s*\1){2,}')
text = REPETITION_PATTERN.sub(r'\1', text)
```

---

## 6. REFERENCES

1. [Deepgram - Whisper Hallucination Study](https://deepgram.com)
2. [Gladia - Whisper-Zero Analysis](https://gladia.io)
3. [OpenAI Whisper GitHub Issues](https://github.com/openai/whisper/issues)
4. [HuggingFace Discussion](https://huggingface.co/openai/whisper-large-v3/discussions)

---

*Document này giải thích tại sao Cherry Core sử dụng Whisper V2 làm default*
