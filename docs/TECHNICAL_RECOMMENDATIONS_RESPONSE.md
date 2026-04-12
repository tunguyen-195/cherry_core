# PHẢN HỒI ĐỀ XUẤT KỸ THUẬT - CHERRY CORE V2

**Ngày:** 2026-01-11
**Phản hồi bởi:** Development Team

---

## 1. TÓM TẮT

Báo cáo đề xuất kỹ thuật chất lượng cao với nhiều gợi ý hữu ích. Phần lớn các đề xuất đã được hoặc sẽ được triển khai.

| Đề xuất | Trạng thái | Ghi chú |
|---------|------------|---------|
| Pyannote 3.1 | ✅ ĐÃ IMPLEMENT | Đang hoạt động |
| SpeechBrain tuning | ✅ ĐÃ IMPLEMENT | Backup engine |
| VBx Refiner improvements | ⚠️ MỘT PHẦN | Sẽ review parameters |
| faster-whisper | 📋 PLANNED | Phase tiếp theo |
| Vietnamese post-processing | 📋 PLANNED | Sẽ implement |
| Word-Speaker Alignment | ✅ ĐÃ IMPLEMENT | AlignmentService |
| Output Formatter | ✅ ĐÃ IMPLEMENT | output_formatter.py |
| Testing | 📋 PLANNED | Sẽ bổ sung |

---

## 2. ĐIỂM ĐÚNG - GHI NHẬN

### 2.1 ✅ Pyannote 3.1 Implementation

**Đề xuất đúng và đã được triển khai.**

File hiện tại: `infrastructure/adapters/diarization/pyannote_adapter.py`

So sánh với đề xuất:

| Feature | Đề xuất | Thực tế | Match? |
|---------|---------|---------|--------|
| Pipeline.from_pretrained | ✅ | ✅ | ✅ |
| use_auth_token | ✅ | ✅ | ✅ |
| num_speakers parameter | ✅ | ✅ | ✅ |
| min/max_speakers | ✅ | ✅ | ✅ |
| CUDA support | ✅ | ⚠️ CPU default | Sẽ cải tiến |

### 2.2 ✅ Vietnamese Post-Processing Dictionary

**Đề xuất rất hữu ích - SẼ IMPLEMENT.**

Danh sách corrections đề xuất:

```python
HOTEL_CORRECTIONS = {
    "đi lặn": "Deluxe",
    "vòng x kế tiếp": "Executive", 
    "điều trú": "lưu trú",
    "cộng phận": "bộ phận",
    "căn cứ công dân": "căn cước công dân",
}
```

**Hành động:** Sẽ tạo file `vietnamese_postprocessor.py` trong Phase tiếp theo.

### 2.3 ✅ Word-Level Alignment với IntervalTree

**Đề xuất hay - Tối ưu hóa performance.**

Hiện tại chúng tôi dùng linear scan O(n*m). IntervalTree sẽ giúp O(n log m).

```python
# Đề xuất
from intervaltree import IntervalTree
speaker_tree = IntervalTree()
for seg in speaker_segments:
    speaker_tree[seg.start_time:seg.end_time] = seg.speaker_id
```

**Hành động:** Sẽ tích hợp trong Phase tối ưu hóa.

### 2.4 ✅ Testing Recommendations

**Đề xuất hợp lý cho production readiness.**

Các test case đề xuất:

- `test_speaker_count_estimation`
- `test_temporal_resolution`
- `test_vbx_refinement`
- `test_end_to_end_transcription`

**Hành động:** Sẽ tạo test suite trong folder `tests/`.

---

## 3. ĐIỂM CẦN LÀM RÕ

### 3.1 ⚠️ VBx loop_prob = 0.75

**Đề xuất:** `loop_prob=0.75` (tăng từ 0.45)

**Phân tích:**

- `loop_prob=0.75` nghĩa là 75% xác suất giữ nguyên speaker
- Transition prob = (1-0.75)/(N-1) = 12.5% (với N=2)
- Điều này làm **GIẢM** khả năng chuyển speaker

**Vấn đề:** Với hội thoại nhanh (tiếng Việt), cần transition cao hơn.

**Kết luận:**

- 0.45 (hiện tại) phù hợp hơn cho hội thoại nhanh
- 0.75 phù hợp cho speeches dài, ít chuyển speaker

Chúng tôi sẽ giữ `loop_prob=0.45` cho use case hiện tại.

### 3.2 ⚠️ SEGMENT_DURATION = 0.5s

**Đề xuất:** Giảm từ 1.2s xuống 0.5s

**Phân tích:**

- 0.5s rất ngắn cho speaker embedding (ECAPA-TDNN)
- Model được train với ~1-3s segments
- Embedding quality giảm đáng kể với segments < 0.8s

**Kết luận:**

- Với Pyannote: Không cần (đã có 16ms resolution)
- Với SpeechBrain: Giữ 1.2s để đảm bảo embedding quality

### 3.3 ⚠️ faster-whisper vs openai-whisper

**Đề xuất:** Chuyển sang faster-whisper

**Đánh giá:**

| Feature | openai-whisper | faster-whisper |
|---------|---------------|----------------|
| Speed | 1x | 4x faster |
| Memory | Higher | Lower |
| Word timestamps | ✅ | ✅ |
| Offline | ✅ | ✅ |
| API stability | ✅ Stable | ⚠️ Active dev |

**Kết luận:** Đây là upgrade hợp lý nhưng không critical. Sẽ implement trong Phase tối ưu hóa.

---

## 4. ĐIỂM KHÁC BIỆT VỚI THỰC TẾ

### 4.1 AlignmentService đã tồn tại

Báo cáo đề xuất tạo `WordSpeakerAligner` class.

**Thực tế:** Chúng tôi đã có `core/services/alignment_service.py` với chức năng tương đương.

Khác biệt chính:

| Feature | Đề xuất | Thực tế |
|---------|---------|---------|
| IntervalTree | ✅ | ❌ Linear scan |
| Nearest speaker fallback | ✅ | ✅ (vừa implement) |
| Word grouping | ✅ | ✅ |

**Kết luận:** Sẽ tích hợp IntervalTree để tối ưu.

### 4.2 OutputFormatter đã tồn tại

Báo cáo đề xuất `OutputFormatter` class.

**Thực tế:** Đã có `core/services/output_formatter.py` với:

- `format_time()`
- `format_subtitle_style()`

Thiếu so với đề xuất:

- `to_srt()` - Format SRT subtitle
- `to_json()` - JSON export

**Hành động:** Sẽ bổ sung các format output.

---

## 5. HÀNH ĐỘNG TIẾP THEO

### Priority 1: Vietnamese Post-Processing

```
[ ] Tạo infrastructure/adapters/correction/vietnamese_postprocessor.py
[ ] Implement HOTEL_CORRECTIONS dictionary
[ ] Integrate vào pipeline
```

### Priority 2: Output Format Extensions

```
[ ] Thêm to_srt() vào OutputFormatter
[ ] Thêm to_json() vào OutputFormatter
```

### Priority 3: Performance Optimization

```
[ ] Tích hợp IntervalTree cho alignment
[ ] Benchmark faster-whisper vs openai-whisper
```

### Priority 4: Testing

```
[ ] Tạo tests/test_diarization.py
[ ] Tạo tests/test_full_pipeline.py
[ ] Setup pytest configuration
```

---

## 6. KẾT LUẬN

Báo cáo đề xuất kỹ thuật **chất lượng cao** với nhiều gợi ý hữu ích.

**Đã triển khai:**

- ✅ Pyannote 3.1 integration
- ✅ Word-level alignment
- ✅ Output formatting

**Sẽ triển khai:**

- 📋 Vietnamese post-processing
- 📋 IntervalTree optimization
- 📋 Test suite
- 📋 faster-whisper (future)

**Giữ nguyên (không đổi):**

- VBx loop_prob=0.45 (phù hợp hội thoại nhanh)
- SEGMENT_DURATION=1.2s (cho SpeechBrain backup)

---

*Phản hồi được viết bởi Development Team*
*Ngày: 2026-01-11*
