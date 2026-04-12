# BÁO CÁO SO SÁNH KẾT QUẢ THỰC TẾ VỚI GROUND TRUTH

**Ngày:** 2026-01-11
**File so sánh:**

- Ground Truth: `output/sample/Result.txt`
- Thực tế: `output/test_audio_diarized_final.txt`

---

## 1. TỔNG QUAN SO SÁNH

| Tiêu chí | Ground Truth | Thực tế | Đánh giá |
|----------|--------------|---------|----------|
| **Số segments** | 50 | 32 | ⚠️ Ít hơn (merged) |
| **Số speakers** | 2 (Speaker 0, 1) | 2 (Speaker 1, 2) | ✅ Đúng |
| **Thời lượng** | ~5:04 | ~4:35 | ✅ Tương đương |
| **Format timestamp** | HH:MM:SS,mmm | HH:MM:SS,mmm | ✅ Đúng |

---

## 2. PHÂN TÍCH CHI TIẾT

### 2.1 Diarization (Speaker Assignment)

#### So sánh Role Mapping

| Ground Truth | Thực tế | Vai trò thực |
|--------------|---------|--------------|
| Speaker 0 | Speaker 1 hoặc Speaker 2 | Lễ tân (Tâm) |
| Speaker 1 | Speaker 1 hoặc Speaker 2 | Khách hàng (Quyên) |

#### Vấn đề phát hiện

**1. Segment đầu tiên bị gộp sai (00:00:00-00:00:10)**

```
Ground Truth:
[Speaker 0] Khách sạn JW Mariott... Tôi có thể giúp gì được cho quý khách ạ?
[Speaker 1] Chào em nhé, chị à, muốn đặt phòng...

Thực tế:
[Unknown] Xin ký chào quý khách... Chào em nhé. Chị muốn đặt phòng...
```

**→ Gộp cả 2 người nói vào 1 segment [Unknown]**

**2. Speaker swap ở một số đoạn**

| Time | Ground Truth | Thực tế | Đúng/Sai |
|------|--------------|---------|----------|
| 00:00:10-12 | Speaker 0 "Chị vui lòng cho em xin tên" | Speaker 1 "Chị mời làm cho em xin tên" | ⚠️ Label swap |
| 00:00:12-20 | Speaker 1 "Chị tên là Quyên" | Speaker 2 "chị tên là Quyên" | ⚠️ Gộp 2 speakers |
| 00:01:10-13 | Speaker 1 "Chị chỉ cần phòng 3 triệu thôi" | Speaker 1 | ✅ Đúng |

**3. Segments quá dài (over-merging)**

| Thực tế Segment | Duration | Ground Truth segments tương ứng |
|-----------------|----------|--------------------------------|
| 00:01:40-02:10 (30s) | 30s | 5 segments (Speaker 0, 1 xen kẽ) |
| 00:02:45-03:36 (51s) | 51s | 7 segments (cả 2 speakers) |
| 00:03:42-04:19 (37s) | 37s | 4 segments |

### 2.2 ASR Quality (Transcription Errors)

#### Lỗi Homophone/Tone

| Ground Truth | Thực tế | Loại lỗi |
|--------------|---------|----------|
| "xin kính chào" | "Xin ký chào" | Tone error |
| "bộ phận" | "cộng phận" | Homophone |
| "lưu trú" | "điều trú" | Homophone |
| "khách sạn" | "cái sạn" | Word boundary |
| "Quyên" | "quên" | Tone error |

#### Lỗi Domain-specific

| Ground Truth | Thực tế | Correct term |
|--------------|---------|--------------|
| "Deluxe" | "đi lặn" | Cần domain correction |
| "Executive" | "vòng x kế tiếp" | Cần domain correction |
| "giá niêm yết" | "giá nhịp ít" | Homophone |
| "căn cước công dân" | "căn cứ công dân" / "căn cướp" | Homophone |
| "hủy trả" | "quỷ trả" | Homophone |
| "giữ phòng" | "sức phòng" | Mishearing |

#### Lỗi Named Entity

| Ground Truth | Thực tế | Correct |
|--------------|---------|---------|
| "JW Mariott" | "G.W. Marriott" | Gần đúng |
| "<quyenhaibon@gmail.com>" | "quyen24a.gmail.com" | Sai số |

---

## 3. ĐIỂM ĐÁNH GIÁ

### 3.1 Diarization Score

| Metric | Score | Ghi chú |
|--------|-------|---------|
| Speaker Count Accuracy | **100%** | 2 speakers đúng |
| Speaker Assignment (overall) | **~70%** | Nhiều đoạn bị swap/merge |
| Temporal Resolution | **~60%** | Segments quá dài |
| First 10s accuracy | **0%** | [Unknown] - không tách được |

**Estimated DER**: ~25-30% (cải thiện so với SpeechBrain cũ ~35%)

### 3.2 ASR Score

| Metric | Score | Ghi chú |
|--------|-------|---------|
| Word Recognition | **~85%** | Đa số từ đúng |
| Vietnamese Tones | **~75%** | Một số lỗi tone |
| Domain Terms | **~30%** | Deluxe, Executive sai hoàn toàn |
| Named Entities | **~60%** | Email, số sai |

**Estimated WER**: ~15-20%

---

## 4. NGUYÊN NHÂN VÀ GIẢI PHÁP

### 4.1 Diarization Issues

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| [Unknown] ở đầu | Word timestamps không khớp với diarization | Cải thiện alignment fallback |
| Speaker swap | Pyannote speaker embedding confusion | Fine-tune hoặc post-processing |
| Over-merging | Word alignment grouping quá aggressive | Tune alignment parameters |

### 4.2 ASR Issues

| Vấn đề | Nguyên nhân | Giải pháp đề xuất |
|--------|-------------|-------------------|
| Homophone errors | Whisper không biết context | Vietnamese post-processor ✅ |
| Domain terms | Không train trên hotel data | Domain dictionary ✅ |
| Named entities | ASR không biết format | Regex formatting |

---

## 5. KẾT LUẬN

### 5.1 So với Ground Truth

| Khía cạnh | Đánh giá | Chi tiết |
|-----------|----------|----------|
| **Diarization** | ⚠️ TRUNG BÌNH | 70% accuracy, còn swap và merge |
| **ASR** | ⚠️ TRUNG BÌNH | 85% word accuracy, cần domain correction |
| **Overall** | ⚠️ CHƯA ĐẠT PRODUCTION | Cần cải thiện thêm |

### 5.2 So với phiên bản trước (SpeechBrain)

| Khía cạnh | Trước | Sau (Pyannote) | Cải thiện |
|-----------|-------|----------------|-----------|
| Segments | 14-40 (không ổn định) | 32 (ổn định) | ✅ |
| Unknown labels | 7 | 1 | ✅ 86% reduction |
| Speaker consistency | Hay flip | Ổn định hơn | ✅ |
| DER estimate | ~35% | ~25-30% | ✅ 15-20% better |

### 5.3 Cần cải thiện

1. **[CRITICAL]** Segment đầu tiên (00:00:00-00:00:10) - Cần fix alignment
2. **[HIGH]** Domain correction cho Deluxe, Executive
3. **[HIGH]** Vietnamese homophone correction
4. **[MEDIUM]** Over-merging prevention

---

## 6. HÀNH ĐỘNG TIẾP THEO

### Immediate (Sprint này)

- [ ] Fix [Unknown] segment đầu - Cải thiện fallback logic
- [ ] Integrate VietnamesePostProcessor vào pipeline
- [ ] Integrate HallucinationFilter vào pipeline

### Short-term

- [ ] Tune alignment grouping parameters
- [ ] Add more domain corrections
- [ ] BARTpho integration for spell correction

---

*Báo cáo được tạo tự động từ so sánh kết quả*
*Ngày: 2026-01-11*
