# ĐỀ XUẤT NGHIÊN CỨU

## Nâng cao độ tin cậy phiên âm bằng giảm hallucination trong mô hình NLP/ASR và xây dựng hệ thống khai phá thông tin từ dữ liệu âm thanh phục vụ trinh sát kỹ thuật

**Ngày:** 2026-03-03  
**Giai đoạn:** Lên ý tưởng & lập kế hoạch (Pre-implementation)  
**Tác giả:** Nhóm nghiên cứu

---

## MỤC LỤC

1. [Đặt vấn đề](#1-đặt-vấn-đề)
2. [Tổng quan hướng tiếp cận](#2-tổng-quan-hướng-tiếp-cận)
3. [Phân tích các thành phần kỹ thuật cần nghiên cứu](#3-phân-tích-các-thành-phần-kỹ-thuật-cần-nghiên-cứu)
4. [Kiến trúc hệ thống đề xuất](#4-kiến-trúc-hệ-thống-đề-xuất)
5. [Roadmap nghiên cứu theo giai đoạn](#5-roadmap-nghiên-cứu-theo-giai-đoạn)
6. [Điểm còn thiếu – Kỹ thuật mới cần bổ sung (2025–2026)](#6-điểm-còn-thiếu--kỹ-thuật-mới-cần-bổ-sung-2025-2026)
7. [Tiêu chí đánh giá thành công](#7-tiêu-chí-đánh-giá-thành-công)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. ĐẶT VẤN ĐỀ

### 1.1 Bối cảnh

Trinh sát kỹ thuật ngày càng phải xử lý khối lượng lớn dữ liệu âm thanh (ghi âm cuộc gọi điện thoại, hội thoại hiện trường, tín hiệu vô tuyến...). Việc phiên âm thủ công tốn nhiều thời gian, dễ sai sót và không thể mở rộng quy mô. Các hệ thống ASR (Automatic Speech Recognition) thương mại như Whisper, Google Speech-to-Text hiện đã đạt độ chính xác cao trên ngôn ngữ phổ biến, nhưng vẫn tồn tại hai rào cản lớn khi áp dụng vào bối cảnh an ninh - tình báo Việt Nam:

1. **Vấn đề hallucination** – Mô hình tự "bịa" ra văn bản không tồn tại trong âm thanh, đặc biệt nguy hiểm khi dùng cho mục đích điều tra (làm giả bằng chứng vô tình).
2. **Ngôn ngữ tiếng Việt** – Đặc thù thanh điệu (6 thanh), phương ngữ vùng miền, từ chuyên ngành nghiệp vụ làm giảm nghiêm trọng độ chính xác của các model nước ngoài.

Ngoài bài toán phiên âm, lực lượng trinh sát còn có nhu cầu **khai phá thông tin** (intelligence mining) từ văn bản: xác định danh tính, địa điểm, thời gian, mối quan hệ, phát hiện mã ngữ và hành vi bất thường trong hội thoại.

### 1.2 Các câu hỏi nghiên cứu

- **RQ1:** Làm thế nào giảm thiểu hallucination trong mô hình ASR/NLP đối với tiếng Việt có giọng vùng miền, tạp âm và chất lượng âm thanh kém?
- **RQ2:** Làm thế nào nhận dạng và phân biệt nhiều người nói (speaker diarization) trong hội thoại tiếng Việt với độ chính xác cao, không phụ thuộc hạ tầng cloud?
- **RQ3:** Làm thế nào tự động khai phá thông tin tình báo từ bản phiên âm: đối tượng, sự kiện, mối liên hệ, mức độ nguy hiểm?
- **RQ4:** Làm thế nào xây dựng pipeline trinh sát kỹ thuật hoàn chỉnh chạy offline, bảo mật, có thể triển khai trong môi trường không có internet?

### 1.3 Tính cấp thiết

- **Thực tiễn**: Lực lượng chức năng cần công cụ phân tích âm thanh nhanh, chính xác, bảo mật.
- **Khoa học**: Tiếng Việt là ngôn ngữ low-resource cho diarization và khai phá thông tin; chưa có hệ thống tích hợp đầy đủ.
- **Độ bảo mật**: Dữ liệu nhạy cảm không thể gửi lên cloud → yêu cầu triển khai local/offline hoàn toàn.

---

## 2. TỔNG QUAN HƯỚNG TIẾP CẬN

### 2.1 Hướng tiếp cận chính

Đề tài nghiên cứu và phát triển **hệ thống AI tích hợp** gồm ba tầng chức năng:

```
┌───────────────────────────────────────────────────────────────────────┐
│  TẦNG 1: TIỀN XỬ LÝ ÂM THANH                                         │
│  VAD + Noise Reduction + Audio Normalization                           │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│  TẦNG 2: NHẬN DẠNG GIỌNG NÓI                                          │
│  ASR (phiên âm tiếng Việt) + Speaker Diarization + Anti-Hallucination │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│  TẦNG 3: KHAI PHÁ THÔNG TIN TÌNH BÁO                                  │
│  LLM Analysis + NER + Relation Extraction + Report Generation          │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 Nguyên tắc thiết kế

1. **Offline-first**: Toàn bộ pipeline phải chạy được không cần internet.
2. **Modular**: Mỗi thành phần là một module độc lập, có thể thay thế/nâng cấp.
3. **Audit-ready**: Mọi output phải có confidence score, nguồn gốc dữ liệu, phục vụ kiểm chứng.
4. **Bảo mật tuyệt đối**: Không ghi log âm thanh gốc ra ngoài, mã hóa kết quả.

---

## 3. PHÂN TÍCH CÁC THÀNH PHẦN KỸ THUẬT CẦN NGHIÊN CỨU

### 3.1 Voice Activity Detection (VAD)

**Vấn đề:** Mô hình ASR hoạt động tệ trên đoạn không có tiếng nói (silence, nhạc nền, nhiễu), sinh ra hallucination.

**Hướng nghiên cứu:**

- Tích hợp **Silero VAD** – mô hình nhẹ (~1.8MB), độ chính xác cao, offline hoàn toàn.
- Cấu hình ngưỡng (threshold) phù hợp tiếng Việt có nhiều khoảng dừng.
- Nghiên cứu kỹ thuật **WebRTC VAD** như phương án dự phòng.

**Chỉ tiêu:** Loại bỏ >95% đoạn không có tiếng nói trước khi đưa vào ASR.

---

### 3.2 Nhận dạng tiếng nói (ASR) tiếng Việt

**Vấn đề:** Các mô hình ASR đa ngôn ngữ (Whisper) chưa được tối ưu cho tiếng Việt với 6 thanh điệu, từ địa phương.

**Các mô hình cần nghiên cứu:**

| Model | Loại | WER Tiếng Việt | Offline? | Ghi chú |
|-------|------|----------------|----------|---------|
| **PhoWhisper-large** | Fine-tuned Whisper | ~10-15% | ✅ | VinAI, 844h dữ liệu VN |
| **Whisper large-v2** | Multilingual | ~15-20% | ✅ | Ít hallucination hơn v3 |
| **Whisper large-v3-turbo** | Multilingual | ~15-20% | ✅ | 6x nhanh hơn v3 |
| **faster-whisper** | CTranslate2 backend | Tương đương | ✅ | 4x nhanh, word timestamps |
| SeamlessM4T-v2 | Meta Universal | Chưa đánh giá | ✅ | Hỗ trợ 100+ ngôn ngữ |

**Hướng nghiên cứu:**

- So sánh độ chính xác và tốc độ các mô hình trên dataset tiếng Việt thực tế.
- Nghiên cứu tác động của Whisper V2 vs V3 đến hallucination rate.
- Đánh giá khả năng fine-tune (LoRA) Whisper trên domain chuyên ngành.

---

### 3.3 Giảm Hallucination trong ASR (TRỌNG TÂM NGHIÊN CỨU #1)

**Vấn đề:** Mô hình Whisper sinh ra văn bản "bịa" trên đoạn im lặng, âm nhạc, hoặc tiếng ồn – đặc biệt nguy hiểm trong ngữ cảnh tư pháp.

**Phân loại hallucination cần nghiên cứu:**

```
Hallucination trong ASR
├── Repetition hallucination: "Quyên. Quyên. Quyên..."
├── Insertion hallucination: Thêm từ không tồn tại ("thanks for watching")
├── Substitution hallucination: Từ sai nghĩa ("JW Marriott" → "G.W. Mirrors")
└── Topic-drift hallucination: Nói sang chủ đề hoàn toàn khác khi gặp silence
```

**Kỹ thuật cần nghiên cứu:**

#### A. Cấu hình Decoding Anti-Hallucination

```python
transcribe(
    condition_on_previous_text=False,  # Tránh lỗi lan truyền
    compression_ratio_threshold=2.0,   # Phát hiện lặp từ
    no_speech_threshold=0.5,           # Từ chối đoạn không có tiếng
    logprob_threshold=-1.0,            # Lọc đầu ra độ tin thấp
)
```

#### B. Bag-of-Hallucinations (BoH) + Delooping

- **Nguồn:** Barański et al., arXiv:2501.11378 (2025)
- **Phương pháp:** Xây dựng từ điển các hallucination phổ biến (tiếng Anh + tiếng Việt), loại bỏ sau khi phiên âm.
- **Kết quả paper:** Giảm 67% erroneous outputs khi kết hợp với VAD.

#### C. Calm-Whisper – Fine-tune Attention Heads

- **Nguồn:** Wang et al., arXiv:2505.12969, Interspeech 2025
- **Phát hiện:** Chỉ 3/20 attention heads trong decoder V3 gây 75% hallucinations ("crazy heads").
- **Phương pháp:** Fine-tune riêng 3 heads này trên dữ liệu non-speech.
- **Kết quả:** ~80% reduction hallucination, <0.1% WER degradation.

#### D. VAD + ASR Pipeline

- Silero VAD loại bỏ đoạn silence trước → ASR không bao giờ xử lý silence.

---

### 3.4 Speaker Diarization (Phân định người nói)

**Vấn đề:** Cần biết "ai nói câu nào" để phân tích vai trò, mối quan hệ trong hội thoại.

**Kiến trúc diarization cần nghiên cứu:**

```
Audio → VAD → Speaker Embedding → Clustering → Diarization Labels
              (ECAPA-TDNN/       (Spectral/
               Wav2Vec2)         Agglomerative)
```

**Các phương pháp cần đánh giá:**

| Phương pháp | Library | DER | Offline? | Ghi chú |
|------------|---------|-----|----------|---------|
| **Pyannote 3.1** | pyannote.audio | ~11% | ⚠️ HF Token | SOTA, cần xác thực online lần đầu |
| **Pyannote 4.0 Community-1** | pyannote.audio | ~9% | ⚠️ HF Token | Cải tiến 17% so với 3.1 |
| **SpeechBrain ECAPA-TDNN** | speechbrain | ~20-25% | ✅ | Hoàn toàn offline |
| **NeMo MSDD** | nemo_toolkit | ~12% | ✅ | NVIDIA, cần GPU A100+ |
| NeMo Sortformer | nemo | ~10% | ✅ | End-to-end transformer |
| WhisperX | whisperx | ~13% | ⚠️ | Tích hợp sẵn diarization |

**Hướng nghiên cứu đặc biệt cho tiếng Việt:**

- Fine-tune Wav2Vec2 trên corpus tiếng Việt cho speaker embeddings (arXiv:2504.18582).
- Nghiên cứu tác động của 6 thanh điệu đến độ chính xác embedding.
- Xây dựng benchmark diarization tiếng Việt: 2, 3, 4, 5+ người nói.

---

### 3.5 Hiệu chỉnh văn bản tiếng Việt (Post-processing)

**Vấn đề:** ASR tiếng Việt mắc nhiều lỗi đồng âm, lỗi tên riêng, lỗi từ chuyên ngành.

**Ví dụ lỗi thực tế:**

```
ASR output: "phòng đi lắc"       → Đúng: "phòng Deluxe"
ASR output: "xá phòng"           → Đúng: "giá phòng"
ASR output: "G.W. Mirrors"       → Đúng: "JW Marriott"
ASR output: "căn cứ công dân"    → Đúng: "căn cước công dân"
ASR output: "Xin ký chào"        → Đúng: "Xin kính chào"
```

**Kỹ thuật cần nghiên cứu:**

| Kỹ thuật | Model | Phù hợp |
|----------|-------|---------|
| Rule-based correction | Dictionary lookup | Từ chuyên ngành, địa danh |
| Seq2Seq correction | BARTpho fine-tuned | Lỗi chính tả tổng quát |
| LLM contextual correction | Qwen/Vistral | Homophones, ngữ cảnh |
| RAG vocabulary | FAISS + embeddings | Domain-specific retrieval |

**BARTpho** (VinAI): Model seq2seq được pre-train trên tiếng Việt, có thể fine-tune cho ASR error correction.

---

### 3.6 Mô hình ngôn ngữ lớn (LLM) cho khai phá thông tin (TRỌNG TÂM NGHIÊN CỨU #2)

**Vấn đề:** Từ bản phiên âm, cần tự động rút trích thông tin tình báo có cấu trúc.

**Thông tin cần khai phá:**

- **5W1H**: Who (Ai?), What (Làm gì?), When (Khi nào?), Where (Ở đâu?), Why (Tại sao?), How (Thế nào?)
- **NER**: Tên người, địa điểm, tổ chức, số điện thoại, số tiền, ngày giờ.
- **Relation extraction**: Mối quan hệ giữa các thực thể.
- **Intent analysis**: Ý định, kế hoạch, mức độ rủi ro.
- **Sentiment/Emotion**: Trạng thái cảm xúc, mức độ khẩn cấp.

**Các mô hình LLM cần đánh giá:**

| Model | Kích thước | Tiếng Việt | Offline? | Ghi chú |
|-------|-----------|-----------|----------|---------|
| **Vistral-7B** | 7B | Tốt | ✅ GGUF | Được fine-tune riêng TV |
| **Qwen2.5-7B** | 7B | Tốt | ✅ GGUF | Alibaba, đa ngôn ngữ |
| **Qwen3-8B** | 8B | Rất tốt | ✅ GGUF | Phiên bản mới nhất |
| **SeaLLM-7B** | 7B | Tốt | ✅ GGUF | Được thiết kế cho Đông Nam Á |
| Gemma-3-9B | 9B | Khá tốt | ✅ GGUF | Google, 2025 |

**Kỹ thuật giảm hallucination cho LLM:**

#### A. Chain-of-Verification (CoVe)

- **Nguồn:** Dhuliawala et al. (2024), ACL Findings.
- **Cơ chế:** LLM tự tạo câu hỏi kiểm tra → trả lời độc lập → cross-check với draft.
- **Kết quả:** Precision tăng 39% trên WikiData, F1 tăng 23% trên MultiSpanQA.

#### B. Constrained Decoding (XGrammar)

- **Vấn đề:** GBNF grammar cho structured JSON output chậm.
- **Giải pháp:** XGrammar (2024) – 10x faster than Outlines, dùng Pushdown Automaton.
- **Tích hợp:** Native trong vLLM v1.0+.

#### C. MEGA-RAG

- Kết hợp Dense retrieval (FAISS) + Sparse (BM25) + Knowledge Graph.
- Giảm 40%+ hallucination trong domain y tế.

---

## 4. KIẾN TRÚC HỆ THỐNG ĐỀ XUẤT

### 4.1 Clean Architecture (Ports & Adapters)

```
cherry_core/
├── core/                        # Domain Layer (pure Python, no deps)
│   ├── domain/
│   │   ├── entities.py          # Transcript, SpeakerSegment, StrategicReport
│   │   └── value_objects.py     # AudioMetadata, ConfidenceScore
│   ├── ports/                   # Abstract interfaces
│   │   ├── i_transcriber.py     # ITranscriber
│   │   ├── i_diarizer.py        # ISpeakerDiarizer
│   │   ├── i_llm_engine.py      # ILLMEngine
│   │   └── i_text_corrector.py  # ITextCorrector
│   └── config.py
│
├── infrastructure/              # Adapter Layer (concrete implementations)
│   └── adapters/
│       ├── asr/                 # PhoWhisper, Whisper V2/V3, FasterWhisper
│       ├── vad/                 # Silero VAD
│       ├── diarization/         # Pyannote, SpeechBrain ECAPA-TDNN
│       ├── llm/                 # LlamaCpp (Vistral/Qwen), vLLM
│       ├── correction/          # BARTpho, dictionary-based
│       ├── alignment/           # Word-Speaker Aligner (IntervalTree)
│       └── rag/                 # FAISS vector store (đề xuất)
│
├── application/                 # Use Cases & Services
│   ├── services/
│   │   ├── transcription_service.py
│   │   ├── analysis_service.py
│   │   ├── cove_service.py      # Chain-of-Verification (đề xuất)
│   │   └── correction_service.py
│   └── use_cases/
│       ├── transcribe_audio.py
│       └── generate_report.py
│
├── presentation/                # Entry Points
│   └── cli/
│
└── prompts/                     # Jinja2 prompt templates
    ├── modules/                 # 5W1H, entities, emotions, SCAN, SVA
    └── scenarios/               # Theo lĩnh vực (khách sạn, an ninh...)
```

### 4.2 Pipeline tích hợp đề xuất

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Audio File (MP3/WAV)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: TIỀN XỬ LÝ                                                    │
│  Silero VAD (threshold=0.3) → Loại đoạn silence                         │
│  ± Noise Reduction (DeepFilterNet tùy chọn)                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ Speech segments only
                               ▼
┌──────────────────────────────┬──────────────────────────────────────────┐
│  STAGE 2A: ASR               │  STAGE 2B: DIARIZATION (Song song)       │
│  PhoWhisper-large            │  Pyannote 4.0 Community-1                │
│  ± faster-whisper            │  ± SpeechBrain ECAPA-TDNN (offline)      │
│  → Word-level timestamps     │  → Speaker labels + timestamps           │
└──────────────────────────────┴──────────────────────────────────────────┘
                               │ Merge via IntervalTree alignment
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ANTI-HALLUCINATION FILTER                                      │
│  1. Delooping (remove "Quyên. Quyên. Quyên...")                         │
│  2. Bag-of-Hallucinations removal (BoH)                                 │
│  3. Compression ratio check (reject if > 2.0)                           │
│  4. Low-confidence segment flagging                                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ Clean diarized transcript
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: POST-PROCESSING TIẾNG VIỆT                                    │
│  1. Domain correction dictionary (hotel, legal, security...)            │
│  2. BARTpho seq2seq correction                                           │
│  3. LLM contextual correction (homophones)                              │
│  4. Phone/number formatting                                              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ Clean speaker-labeled transcript
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: KHAI PHÁ THÔNG TIN (LLM Intelligence Mining)                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Draft Report (Vistral-7B / Qwen3-8B + GBNF/XGrammar)           │    │
│  │  → 5W1H, NER, Timeline, Risk Assessment, Intent Analysis        │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │ Chain-of-Verification (CoVe)             │
│  ┌───────────────────────────▼─────────────────────────────────────┐    │
│  │  Verification Questions → Independent Answers → Cross-check      │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐    │
│  │  Final Structured Report (JSON + Human-readable)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: JSON + TXT + SRT                                               │
│  - Speaker-labeled transcript (diarized)                                │
│  - Named entities (persons, locations, organizations, dates)            │
│  - Risk indicators & intent flags                                       │
│  - Timeline of events                                                   │
│  - Strategic analysis report                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. ROADMAP NGHIÊN CỨU THEO GIAI ĐOẠN

### Giai đoạn 1: Nền tảng & Môi trường (Tuần 1-2)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| Thiết lập môi trường | Python 3.10+, CUDA, FFmpeg, venv | Environment ready |
| Tải & đánh giá mô hình | PhoWhisper, Whisper V2/V3, Pyannote | Model benchmarks |
| Xây dựng Clean Architecture | Core ports & adapters skeleton | Codebase structure |
| Thu thập dataset | Ghi âm thực tế, VIVOS, VLSP 2020 | Test corpus |

**Milestone:** Có thể chạy phiên âm cơ bản với PhoWhisper.

---

### Giai đoạn 2: Phiên âm & Chống Hallucination (Tuần 3-5)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| Tích hợp Silero VAD | Preprocessing pipeline | VAD module |
| Triển khai anti-hallucination configs | Decoding params | Reduced WER |
| Xây dựng BoH filter cho tiếng Việt | Vietnamese hallucination dictionary | BoH module |
| Implement Delooping | Regex + phrase-level | Deloop module |
| Benchmark ASR | WER trên VIVOS, VLSP, audio thực tế | WER report |

**Milestone:** WER < 15% trên audio thực tế, hallucination rate giảm > 50%.

---

### Giai đoạn 3: Speaker Diarization (Tuần 6-8)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| Tích hợp Pyannote 3.1 | Primary diarization engine | Pyannote adapter |
| Tích hợp SpeechBrain ECAPA-TDNN | Offline fallback | SpeechBrain adapter |
| Xây dựng Word-Speaker Aligner | IntervalTree-based alignment | Alignment module |
| Benchmark diarization | DER trên các scenarios | DER report |
| Tune clustering params | Silhouette analysis | Optimized config |

**Milestone:** DER < 15% trên hội thoại 2 người, < 20% trên hội thoại 3-4 người.

---

### Giai đoạn 4: Post-Processing tiếng Việt (Tuần 9-10)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| Domain correction dictionary | Hotel, legal, security terms | Vocabulary files |
| BARTpho integration | Seq2seq correction | BARTpho adapter |
| Multi-pass correction pipeline | Rule → Model → LLM | Correction service |
| Benchmark correction | Precision/Recall trên error corpus | Correction report |

**Milestone:** Correction accuracy > 80% trên lỗi phonetic phổ biến.

---

### Giai đoạn 5: Khai phá thông tin (Tuần 11-14)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| Tích hợp Vistral-7B / Qwen3-8B | LLM via llama.cpp | LLM adapter |
| Thiết kế prompt templates | 5W1H, NER, risk assessment | Prompt library |
| Implement CoVe | Chain-of-Verification service | CoVe service |
| Constrained decoding | XGrammar / GBNF | Structured JSON output |
| Scientific frameworks | SVA, SCAN analysis | Analysis modules |

**Milestone:** Báo cáo tình báo tự động đạt độ chính xác > 85% so với annotation chuyên gia.

---

### Giai đoạn 6: Tích hợp, Kiểm thử & Đánh giá (Tuần 15-18)

| Nhiệm vụ | Mô tả | Output |
|----------|-------|--------|
| End-to-end integration testing | Full pipeline test | Integration test suite |
| Benchmark đa dạng | Nhiều điều kiện audio | Generalization report |
| Performance optimization | Batching, GPU optimization | Optimized pipeline |
| Security audit | Dữ liệu không rò rỉ | Security checklist |
| Viết báo cáo khoa học | Papers + thesis | Publications |

---

## 6. ĐIỂM CÒN THIẾU – KỸ THUẬT MỚI CẦN BỔ SUNG (2025-2026)

### 6.1 Những kỹ thuật mới nhất chưa được đề cập đủ

#### A. End-to-End Neural Diarization (EEND-EDA)

**Vấn đề với pipeline cascaded hiện tại:** ASR và Diarization chạy riêng biệt → lỗi ở bước trước ảnh hưởng bước sau.

**EEND-EDA** (End-to-End Neural Diarization with Encoder-Decoder Attractors):

- Diarize trực tiếp từ audio features mà không cần clustering riêng.
- Xử lý tự nhiên overlapping speech (nói đè lên nhau).
- Không cần biết số lượng người nói trước.
- **Nguồn:** Horiguchi et al. (2022, 2023), NeurIPS – vẫn là hướng nghiên cứu tích cực 2025.

> **Khuyến nghị:** Nghiên cứu tích hợp EEND-EDA làm phương án thay thế cho pipeline Pyannote + Clustering hiện tại.

---

#### B. VietASR – ASR Hiệu Quả Cực Cao Cho Tiếng Việt

- **Nguồn:** arXiv:2505.21527 (2025)
- Đạt chất lượng industry-level chỉ với **50 giờ** dữ liệu labeled.
- Kết hợp Semi-supervised learning + Data augmentation.
- Tiềm năng fine-tune trên dữ liệu chuyên ngành an ninh.

> **Khuyến nghị:** Nghiên cứu phương pháp VietASR và khả năng áp dụng để fine-tune với dữ liệu ghi âm nghiệp vụ (ít nhất 50h).

---

#### C. Streaming ASR / Real-time Diarization

**Hạn chế hiện tại:** Pipeline xử lý batch (file hoàn chỉnh) → không hỗ trợ real-time.

**Kỹ thuật cần nghiên cứu:**

- **Whisper Streaming** (Peng et al., 2023) – tích hợp Whisper với LocalAgreement cho online ASR.
- **Online Speaker Diarization** – xử lý theo chunk audio thời gian thực.
- Ứng dụng thực tiễn: Giám sát âm thanh trực tiếp, phát hiện từ khóa tức thời.

> **Khuyến nghị:** Nghiên cứu thêm module streaming ASR như Phase 7 sau khi hoàn thành pipeline batch.

---

#### D. Audio Forensics – Phát Hiện Giọng Tổng Hợp (Deepfake Voice)

**Vấn đề mới nổi 2024-2025:** Các công cụ như ElevenLabs, RVC có thể tạo giọng giả rất chân thực → nguy cơ làm giả bằng chứng âm thanh.

**Kỹ thuật cần nghiên cứu:**

- **Anti-spoofing models**: AASIST, RawBoost (SOTA 2024 trong ASVspoof challenge).
- **Watermarking detection**: Phát hiện watermark trong audio AI-generated.
- Tích hợp module phân loại: Natural vs Synthetic speech trước khi phiên âm.

> **Khuyến nghị:** Đây là hướng nghiên cứu quan trọng cho bối cảnh trinh sát, cần có module phát hiện deepfake voice trong pipeline.

---

#### E. Multi-Modal Intelligence Extraction

**Hạn chế hiện tại:** Chỉ xử lý audio → text → analysis.

**Hướng mở rộng:**

- Kết hợp phân tích âm thanh (prosody, tempo, stress pattern) với text.
- **Paralinguistic analysis:** Phát hiện stress, lying indicators qua giọng nói.
- **Cross-modal verification:** So sánh nội dung nói với thông tin từ nguồn khác.

> **Khuyến nghị:** Nghiên cứu thêm acoustic feature extraction (librosa) bổ sung cho LLM text analysis.

---

#### F. Federated Learning / Privacy-Preserving Fine-tuning

**Vấn đề:** Dữ liệu âm thanh nhạy cảm – không thể upload lên server tập trung để fine-tune model.

**Kỹ thuật:**

- **Federated Learning:** Fine-tune model phân tán không cần chia sẻ dữ liệu gốc.
- **Differential Privacy:** Đảm bảo fine-tune không leak thông tin cá nhân.

> **Khuyến nghị:** Cần nghiên cứu cơ chế fine-tune an toàn trong môi trường phân loại, đặc biệt khi mở rộng ra nhiều đơn vị sử dụng.

---

#### G. Confidence Calibration và Uncertainty Quantification

**Vấn đề:** Mô hình ASR/LLM thường overconfident – confidence score cao nhưng vẫn sai.

**Kỹ thuật cần nghiên cứu:**

- Temperature scaling cho LLM uncertainty.
- **Token-level confidence** từ Whisper – đánh dấu từ có độ tin thấp.
- Kết hợp nhiều mô hình (ensemble) để estimate uncertainty.

> **Khuyến nghị:** Mọi output cần kèm confidence score được calibrate đúng, đặc biệt quan trọng cho báo cáo tư pháp.

---

#### H. Semantic Knowledge Graph cho Relation Extraction

**Hạn chế hiện tại:** LLM extract thông tin nhưng không structure thành knowledge graph.

**Kỹ thuật:**

- **Neo4j + spaCy**: Xây dựng knowledge graph từ transcript.
- **Entity-Relation extraction**: Fine-tune model NER+RE trên tiếng Việt.
- Khai phá mạng lưới quan hệ giữa các đối tượng qua nhiều cuộc gọi.

> **Khuyến nghị:** Research thêm phần này như Phase 8 – phân tích liên cuộc gọi, mạng lưới đối tượng.

---

### 6.2 Bảng tổng hợp khoảng trống và đề xuất

| # | Khoảng trống | Kỹ thuật đề xuất | Độ ưu tiên | Effort | Timeline |
|---|-------------|-----------------|-----------|--------|---------|
| 1 | Overlapping speech xử lý kém | EEND-EDA end-to-end diarization | 🔴 Cao | 3-4 tuần | Phase 3+ |
| 2 | Fine-tune ASR với ít data | VietASR semi-supervised | 🔴 Cao | 2-3 tuần | Phase 2+ |
| 3 | Không hỗ trợ real-time | Whisper Streaming | 🟡 TB | 2 tuần | Phase 7 |
| 4 | Không phát hiện deepfake voice | AASIST anti-spoofing | 🔴 Cao | 1-2 tuần | Phase 2 |
| 5 | Confidence không calibrate | Token-level uncertainty | 🟡 TB | 1 tuần | Phase 5 |
| 6 | Không có knowledge graph | Neo4j + relation extraction | 🟢 Thấp | 3-4 tuần | Phase 8 |
| 7 | Privacy trong fine-tuning | Federated Learning / DP | 🟡 TB | 4+ tuần | Future |
| 8 | Chỉ phân tích text, không phân tích acoustic | Paralinguistic features | 🟡 TB | 2 tuần | Phase 6 |

---

## 7. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG

### 7.1 Chỉ số kỹ thuật

| Chỉ số | Baseline (Dự kiến) | Mục tiêu |
|--------|-------------------|---------|
| WER tiếng Việt | ~20-25% | < 10% |
| Hallucination rate | Chưa đo | < 3% |
| Speaker DER | N/A | < 15% (2 người), < 20% (3-4 người) |
| Correction accuracy | ~60% | > 85% |
| LLM JSON valid rate | ~90% | 100% |
| LLM factual accuracy | Chưa đo | > 85% so với expert annotation |
| Deepfake detection EER | N/A | < 5% |
| Processing speed | N/A | ≤ 2x realtime trên GPU |

### 7.2 Chỉ số thực tiễn

- **Offline hoàn toàn**: Pipeline chạy không cần internet sau khi setup.
- **Bảo mật**: Zero external data leakage.
- **Độ tin cậy pháp lý**: Output có đủ metadata để làm tài liệu tham chiếu.
- **Khả mở rộng**: Dễ thêm domain mới (y tế, tài chính, pháp lý).

---

## 8. TÀI LIỆU THAM KHẢO

### ASR & Hallucination

1. Le D.T. et al. (2024). "PhoWhisper: Automatic Speech Recognition for Vietnamese." *ICLR 2024 Tiny Papers*. [arXiv:2406.02555](https://arxiv.org/abs/2406.02555)

2. Barański M. et al. (2025). "Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio." [arXiv:2501.11378](https://arxiv.org/abs/2501.11378)

3. Wang J. et al. (2025). "Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down." *Interspeech 2025*. [arXiv:2505.12969](https://arxiv.org/abs/2505.12969)

4. Deepgram Research (2024). "Whisper-v3 Hallucinations on Real World Data." [deepgram.com](https://deepgram.com/learn/whisper-v3-results)

5. Tran et al. (2025). "VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data." [arXiv:2505.21527](https://arxiv.org/abs/2505.21527)

6. Pham B.T. et al. (2024). "Enhancing Whisper Model for Vietnamese Specific Domain with Data Blending and LoRA Fine-Tuning." *ICISN 2024*. [SpringerLink](https://link.springer.com/chapter/10.1007/978-981-97-5504-2_18)

### Speaker Diarization

1. Bredin H. et al. (2023). "Pyannote.audio 2.1 Speaker Diarization Pipeline." *ICASSP 2023*. [arXiv:2212.02060](https://arxiv.org/abs/2212.02060)

2. Horiguchi S. et al. (2022). "End-to-End Speaker Diarization as Post-Processing." *ICASSP 2022*.

3. Vo T.H. et al. (2025). "Speaker Diarization for Low-Resource Languages Through Wav2Vec Fine-Tuning." [arXiv:2504.18582](https://arxiv.org/abs/2504.18582)

4. Nguyen P.T. et al. (2022). "Speaker Diarization For Vietnamese Conversations Using Deep Neural Network Embeddings." *IEEE Xplore*. [DOI:10.1109/ICSEC56439.2022.9852042](https://ieeexplore.ieee.org/document/9852042)

5. Bain M. et al. (2023). "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio." [GitHub](https://github.com/m-bain/whisperX)

### LLM & Knowledge Extraction

1. Dhuliawala S. et al. (2024). "Chain-of-Verification Reduces Hallucination in Large Language Models." *ACL 2024 Findings*. [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)

2. Dong M. et al. (2024). "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models." *Preprint*.

3. Zhang T. et al. (2025). "MEGA-RAG: Multi-Evidence Guided Answer Refinement for Mitigating Hallucinations." *PMC 2025*.

### Anti-Spoofing / Deepfake Detection

1. Jung J.W. et al. (2022). "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." *ICASSP 2022*.

2. Yi J. et al. (2022). "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing." *ICASSP 2022*.

### Vietnamese NLP

1. Nguyen L.T. et al. (2021). "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese." [arXiv:2109.09701](https://arxiv.org/abs/2109.09701)

2. Bui M.D. et al. (2024). "Fine-tuned BARTpho for Vietnamese ASR Correction." [GitHub](https://github.com/bmd1905/vietnamese-correction)

3. Tran T.T. et al. (2021). "VSEC: Transformer-based Model for Vietnamese Spelling Correction." [arXiv:2111.00640](https://arxiv.org/abs/2111.00640)

---

*Tài liệu này là bản đề xuất ý tưởng nghiên cứu ban đầu, phục vụ thảo luận và xin ý kiến chuyên gia trước khi tiến hành triển khai.*

*Ngày lập: 2026-03-03 | Phiên bản: 1.0 | Trạng thái: Draft*
