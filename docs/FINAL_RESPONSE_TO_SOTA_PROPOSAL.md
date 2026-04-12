# PHẢN HỒI CUỐI CÙNG - SOTA UPGRADE PROPOSAL

**Ngày:** 2026-01-11
**Phản hồi bởi:** Development Team

---

## 1. ĐIỂM ĐỒNG THUẬN HOÀN TOÀN

### 1.1 ✅ Whisper V3 Hallucination - XÁC NHẬN ĐÚNG

**Kinh nghiệm của User:** "V3 hay bị ảo giác, kết quả không tốt bằng V2"

**Research xác nhận:**
> "Whisper Large-v3 hallucinates **4 times more frequently** than v2, showing a median WER of 53.4 compared to v2's 12.7 on real-world data."
> — [Deepgram Research, 2024]

| Metric | Whisper V2 | Whisper V3 |
|--------|------------|------------|
| Hallucination frequency | 1x (baseline) | **4x worse** |
| Repetition issues | Có | **Tệ hơn** |
| WER (real-world) | 12.7 | 53.4 |

**Nguyên nhân V3 hallucinate:**

1. Training data noisy hơn
2. Model predict next word thay vì transcribe
3. Bias từ YouTube outros ("Thank you for watching")
4. Silent segments gây trigger hallucination

**Kết luận:** User hoàn toàn đúng. Giữ Whisper V2 là quyết định sáng suốt.

**Hành động:** Giữ nguyên Whisper V2 làm default. Chỉ dùng V3-turbo nếu có VAD preprocessing mạnh.

---

### 1.2 ✅ Pyannote Community-1 - XÁC NHẬN CẢI TIẾN

**Research xác nhận** (từ HuggingFace benchmark):

| Dataset | 3.1 DER | Community-1 DER | Cải thiện |
|---------|---------|-----------------|-----------|
| AISHELL-4 | 12.2% | 11.7% | -4% |
| AliMeeting | 24.5% | 20.3% | **-17%** |
| AMI IHM | 18.8% | 17.0% | -10% |
| AMI SDM | 22.7% | 19.9% | -12% |
| DIHARD 3 | 21.4% | 20.2% | -6% |

**Cải tiến chính:**

- Giảm đáng kể **speaker confusion** (gán sai speaker)
- Better **speaker counting** accuracy
- Simpler offline use

**Hành động:** SẼ UPGRADE lên Community-1. Đây là upgrade dễ dàng:

```python
# OLD
"pyannote/speaker-diarization-3.1"
# NEW
"pyannote/speaker-diarization-community-1"
```

---

### 1.3 ✅ Generalization Testing - GHI NHẬN

Đề xuất về benchmark suite với diverse audio là **hoàn toàn hợp lý**.

**Rủi ro hiện tại:**

- Chỉ test với 1 file: `test_audio.mp3` (hotel booking, 2 speakers, studio quality)
- Có thể overfit cho trường hợp này

**Hành động:** Sẽ tạo benchmark suite với:

- [ ] 2 speakers (hotel) - hiện có
- [ ] 3-4 speakers (meeting)
- [ ] Phone call quality (8kHz)
- [ ] Noisy environment
- [ ] Overlapping speech

---

### 1.4 ✅ WhisperX - ĐÁNG XEM XÉT

WhisperX là giải pháp integrated tốt nhưng cần đánh giá:

**Ưu điểm:**

- All-in-one pipeline
- wav2vec2 alignment (chính xác hơn)
- Battle-tested

**Nhược điểm:**

- Thêm dependencies
- Khó debug nếu có lỗi
- Phụ thuộc vào 3rd party maintenance

**Hành động:** Đánh giá trong Phase sau, không urgent.

---

## 2. ĐIỂM CẦN LÀM RÕ

### 2.1 ⚠️ VinAI Grammar API

**Đề xuất:** Thay ProtonX bằng VinAI Grammar API

**Vấn đề:**

- VinAI Grammar API là **online service**
- Không phù hợp với mục tiêu **offline** của Cherry Core
- BARTpho có thể chạy offline nhưng cần GPU memory cao

**Hành động:** Giữ domain-specific correction dictionary (hiện tại) + xem xét BARTpho cho offline.

### 2.2 ⚠️ Hardcoded Parameters "Overfitting"

**Đề xuất:** Các params như `threshold=0.3`, `min_speech_duration_ms=100` có thể gây overfit.

**Phản biện:**

- Các giá trị này là **industry defaults**, không phải tự đặt
- Silero VAD khuyến nghị threshold 0.3-0.5
- 100ms min speech là hợp lý (< 100ms không phải speech thật)

**Kết luận:** Không phải overfitting, đây là reasonable defaults.

---

## 3. KẾ HOẠCH NÂNG CẤP CHÍNH THỨC

### Sprint 1 (Immediate)

| Task | Priority | Status |
|------|----------|--------|
| Upgrade Pyannote 3.1 → Community-1 | **CRITICAL** | 📋 TODO |
| Document Whisper V3 hallucination issue | HIGH | 📋 TODO |
| Giữ Whisper V2 làm default | HIGH | ✅ DONE |

### Sprint 2 (Short-term)

| Task | Priority | Status |
|------|----------|--------|
| Tạo benchmark suite đa dạng audio | HIGH | 📋 TODO |
| Run generalization tests | HIGH | 📋 TODO |
| Integrate IntervalTree | MEDIUM | 📋 TODO |

### Sprint 3 (Medium-term)

| Task | Priority | Status |
|------|----------|--------|
| Evaluate WhisperX | MEDIUM | 📋 TODO |
| Evaluate faster-whisper V3-turbo + VAD | MEDIUM | 📋 TODO |
| Complete test suite | MEDIUM | 📋 TODO |

---

## 4. TÓM TẮT

### Điểm ĐỒNG Ý hoàn toàn

1. ✅ Whisper V3 hallucination - User đúng, giữ V2
2. ✅ Pyannote Community-1 upgrade - Sẽ implement
3. ✅ Generalization testing - Cần benchmark suite
4. ✅ WhisperX - Đáng xem xét cho future

### Điểm KHÔNG ĐỒNG Ý

1. ❌ VinAI Grammar API - Không offline, không phù hợp
2. ❌ Parameters "overfitting" - Đây là industry defaults

### Điểm ĐÃ IMPLEMENT

1. ✅ Pyannote 3.1 - Đang chạy
2. ✅ Vietnamese post-processor - Vừa tạo
3. ✅ Whisper V2 default - Đã có

---

## 5. REFERENCES

### Research Sources

1. [Deepgram - Whisper Hallucination Study](https://deepgram.com) - V3 hallucinates 4x more
2. [Gladia - Whisper Hallucination Analysis](https://gladia.io) - Training bias causes issues
3. [Pyannote Community-1 Release](https://www.pyannote.ai/blog/community-1) - Official benchmark
4. [HuggingFace Pyannote Models](https://huggingface.co/pyannote) - DER comparisons

---

*Phản hồi cuối cùng bởi Development Team*
*Dựa trên research và evidence*
*Ngày: 2026-01-11*
