# Báo Cáo Nghiên Cứu Nâng Cấp Cherry Core V2
## Giảm Ảo Giác & Tăng Độ Chính Xác

**Ngày**: 2026-01-11  
**Phiên bản**: 1.0  
**Tác giả**: AI Research Agent

---

## Mục Lục
1. [Tóm Tắt Dự Án](#1-tóm-tắt-dự-án)
2. [Phân Tích Hiện Trạng](#2-phân-tích-hiện-trạng)
3. [Nghiên Cứu Kỹ Thuật Giảm Ảo Giác](#3-nghiên-cứu-kỹ-thuật-giảm-ảo-giác)
4. [Nghiên Cứu Speaker Diarization](#4-nghiên-cứu-speaker-diarization)
5. [Đề Xuất Nâng Cấp](#5-đề-xuất-nâng-cấp)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Tài Liệu Tham Khảo](#7-tài-liệu-tham-khảo)

---

## 1. Tóm Tắt Dự Án

### 1.1 Mục Tiêu Cherry Core V2
**Cherry Core V2** là hệ thống **Forensic Audio Intelligence** phục vụ điều tra tội phạm với 4 pipeline chính:

1. **Transcription (ASR)**: Chuyển audio tiếng Việt thành text
   - Models: PhoWhisper, Whisper V2/V3
   - Anti-hallucination configs đã có

2. **Spelling Correction**: Sửa lỗi chính tả/ngữ âm
   - ProtonX Seq2Seq (grammar/spelling)
   - LLM Contextual Correction (homophones)

3. **Forensic Analysis**: Phân tích tình báo
   - Vistral-7B với GBNF grammar
   - Scientific frameworks: SVA, SCAN, Psychology VN

4. **Structured Report**: Báo cáo JSON chuẩn hóa

### 1.2 Kiến Trúc Hiện Tại
```
cherry_core/
├── core/                    # Domain Layer
│   ├── domain/entities.py   # Transcript, StrategicReport
│   └── ports/               # ITranscriber, ILLMEngine, ITextCorrector
├── infrastructure/          # Adapter Layer
│   ├── adapters/
│   │   ├── asr/             # WhisperV2Adapter (Champion)
│   │   ├── llm/             # LlamaCppAdapter, VLLMAdapter
│   │   ├── correction/      # ProtonXAdapter
│   │   └── diarization/     # ResemblyzerAdapter (chưa integrate)
│   └── factories/           # SystemFactory (DI)
├── application/             # Application Layer
│   ├── services/            # AnalysisService, CorrectionService
│   └── use_cases/           # TranscribeAudio, GenerateReport
└── prompts/                 # Jinja2 Templates, Scenarios, GBNF
```

---

## 2. Phân Tích Hiện Trạng

### 2.1 Điểm Mạnh

| Component | Implementation | Status |
|-----------|---------------|--------|
| Hexagonal Architecture | Ports & Adapters pattern | ✅ Tốt |
| ASR Anti-Hallucination | `condition_on_previous_text=False`, thresholds | ✅ Có |
| Repetition Filter | Regex-based `_remove_repetitions()` | ✅ Có |
| GBNF Grammar | `json_schema.gbnf` cho structured output | ⚠️ Disabled |
| Vocabulary Injection | 73k+ từ, keyword-based retrieval | ⚠️ Cần semantic |
| Scientific Prompts | SVA, SCAN, Psychology VN modules | ✅ Tốt |
| Speaker Diarization | `ResemblyzerAdapter` created | ⚠️ Chưa integrate |

### 2.2 Vấn Đề Phát Hiện Từ Output Thực Tế

#### A. Lỗi ASR Phonetic (Từ transcript thực tế)
```
Raw:  "tiện xương hô"     → Expected: "tiện xưng hô"
Raw:  "phòng đi lắc"      → Expected: "phòng Deluxe"  
Raw:  "xkz"               → Expected: "Executive"
Raw:  "G.W.Mirrors"       → Expected: "GW Marriott"
Raw:  "xá phòng"          → Expected: "giá phòng"
```

**Nguyên nhân**: Whisper không có Vietnamese phoneme model, dẫn đến nhầm lẫn các từ đồng âm.

#### B. LLM Correction Bỏ Sót
- "xá phòng" không được sửa thành "giá phòng" (ngữ cảnh về tiền)
- Vocabulary trigger quá generic, không bắt được context

#### C. GBNF Grammar Disabled
- Lý do: Chậm khi kết hợp với llama.cpp
- Hệ quả: JSON output đôi khi invalid

#### D. Speaker Diarization Chưa Tích Hợp
- `ResemblyzerAdapter` đã tạo nhưng chưa dùng
- Transcript không có speaker labels → LLM khó phân tích vai trò

### 2.3 Metrics Baseline

| Metric | Ước tính hiện tại | Nguồn đánh giá |
|--------|-------------------|----------------|
| ASR WER (Vietnamese) | ~15-20% | So sánh raw vs corrected |
| JSON Parse Success | ~95% | Logs từ AnalysisService |
| Hallucination Rate | Unknown | Cần benchmark |
| Homophone Correction | ~70% | Sample "xương hô" → "xưng hô" |

---

## 3. Nghiên Cứu Kỹ Thuật Giảm Ảo Giác

### 3.1 Phân Loại Hallucination

Theo survey MDPI 2025 "From Illusion to Insight", hallucination mitigation có 6 categories:

1. **Training and Learning Approaches**: Fine-tuning, RLHF, Knowledge Distillation
2. **Architectural Modifications**: Attention mechanisms, Memory augmentation
3. **Input/Prompt Optimization**: Prompt engineering, Few-shot learning
4. **Post-Generation Quality Control**: Self-verification, Fact-checking
5. **Interpretability and Diagnostic Methods**: Confidence scoring
6. **Agent-Based Orchestration**: Multi-agent debate, Tool augmentation

### 3.2 Chain-of-Verification (CoVe) - Kỹ Thuật Đề Xuất Chính

**Paper**: Dhuliawala et al. (2024). "Chain-of-Verification Reduces Hallucination in Large Language Models". ACL Findings.

#### Cơ Chế Hoạt Động
```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Draft Response                                      │
│ - LLM tạo báo cáo ban đầu (có thể chứa hallucination)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Verification Planning                               │
│ - LLM tạo danh sách câu hỏi kiểm tra facts trong draft     │
│ - VD: "Người gọi tên gì?", "Giá phòng bao nhiêu?"          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Independent Verification (CRITICAL)                 │
│ - Trả lời TỪNG câu hỏi RIÊNG BIỆT (không context draft)    │
│ - Tránh confirmation bias                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Final Synthesis                                     │
│ - Cross-check draft vs verification answers                 │
│ - Sửa inconsistencies → Final Report                        │
└─────────────────────────────────────────────────────────────┘
```

#### Kết Quả Benchmark (từ paper)
- **Wikidata List Questions**: Precision tăng từ 0.51 → 0.71 (+39%)
- **MultiSpanQA**: F1 score tăng từ 0.39 → 0.48 (+23%)
- **Longform Biography**: FACTSCORE tăng 28%

#### Áp Dụng Cho Cherry Core
```python
# application/services/cove_service.py (proposed)
class ChainOfVerificationService:
    def analyze_with_verification(self, transcript: str) -> dict:
        # Step 1: Draft
        draft = self.llm.generate(draft_prompt.format(transcript=transcript))
        
        # Step 2: Plan verification questions
        questions = self.llm.generate(verification_plan_prompt.format(
            transcript=transcript, draft=draft
        ))
        
        # Step 3: Answer independently (CRITICAL: no draft context)
        answers = []
        for q in questions:
            ans = self.llm.generate(verification_answer_prompt.format(
                transcript=transcript, question=q
            ))  # NOTE: draft không được truyền vào
            answers.append(ans)
        
        # Step 4: Synthesize
        final = self.llm.generate(synthesis_prompt.format(
            draft=draft, verifications=zip(questions, answers)
        ))
        return final
```

### 3.3 Whisper Hallucination Mitigation

#### Nguyên Nhân Whisper Hallucination
Theo paper "Calm-Whisper" (2025):
- 1% transcriptions chứa hallucinated phrases
- 38% trong số đó có explicit harms
- Xảy ra chủ yếu ở **non-speech segments** (silence, noise)
- 3/20 attention heads chịu trách nhiệm 75% hallucinations

#### Kỹ Thuật Hiện Có Trong Dự Án
```python
# infrastructure/adapters/asr/whisperv2_adapter.py
result = self.model.transcribe(
    audio_path,
    language="vi",
    beam_size=5,
    best_of=5,
    temperature=0.0,
    condition_on_previous_text=False,  # ✅ Prevent cascading errors
    compression_ratio_threshold=2.0,    # ✅ Detect repetition
    logprob_threshold=-1.0,             # ✅ Confidence filter
    no_speech_threshold=0.5,            # ✅ Silence detection
)
```

#### Đề Xuất Bổ Sung
1. **VAD Preprocessing**: Silero VAD để loại bỏ silence trước khi đưa vào Whisper
2. **N-gram Penalty**: `no_repeat_ngram_size` trong decoding
3. **Post-processing Enhancement**: Regex patterns cho Vietnamese-specific hallucinations

### 3.4 RAG (Retrieval-Augmented Generation)

#### Nghiên Cứu MEGA-RAG (2025)
- Kết hợp 3 retrieval methods: Dense (FAISS), Sparse (BM25), Knowledge Graph
- Cross-encoder reranking cho relevance
- **Giảm 40%+ hallucination rate** trong medical domain

#### Áp Dụng Cho Cherry Core
Hiện tại dự án dùng keyword matching cho vocabulary:
```python
# application/services/correction_service.py
def _get_relevant_vocab(self, transcript: str) -> str:
    hospitality_triggers = ["khách sạn", "đặt phòng", ...]
    if any(t in transcript_lower for t in hospitality_triggers):
        relevant_terms.extend(categories["hospitality"][:30])
```

**Đề xuất**: Chuyển sang embedding-based retrieval với sentence-transformers.

### 3.5 Constrained Decoding (GBNF/XGrammar)

#### Vấn Đề Với GBNF Hiện Tại
- GBNF trong llama.cpp chậm do FSM token-by-token
- Đã disable trong dự án vì performance

#### Giải Pháp: XGrammar (2024)
- Dùng Pushdown Automaton thay FSM
- Batch constrained decoding
- Grammar compilation trong C (không phải Python)
- **10x faster** so với Outlines
- Đã tích hợp native trong vLLM

---

## 4. Nghiên Cứu Speaker Diarization

### 4.1 Tại Sao Cần Diarization?

Từ transcript hiện tại:
```
"Chào em nhé. Chị muốn đặt phòng ở bên khách sạn mình í. 
Em giúp chị với. Chị vui lòng cho em xin tên để em tiện xưng hô với ạ."
```

**Vấn đề**: Không biết ai nói câu nào → LLM khó phân tích:
- Ai là khách? Ai là nhân viên?
- Ai đưa ra yêu cầu? Ai xác nhận?

**Mong muốn**:
```
[SPEAKER_1 - Khách]: Chào em nhé. Chị muốn đặt phòng...
[SPEAKER_2 - Lễ tân]: Chị vui lòng cho em xin tên...
```

### 4.2 Hiện Trạng Trong Dự Án

#### Đã Có
```python
# infrastructure/adapters/diarization/resemblyzer_adapter.py
class ResemblyzerAdapter(ISpeakerDiarizer):
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        # 1. Load audio với preprocess_wav
        # 2. Split thành 1-second segments
        # 3. Compute d-vector embeddings
        # 4. Spectral Clustering
        # 5. Return labeled segments
```

#### Chưa Có
- Integration với `TranscriptionService`
- Alignment ASR timestamps với diarization segments
- Speaker-aware prompts trong analysis

### 4.3 Các Phương Pháp Diarization

| Approach | Library | Accuracy | Offline? | GPU? | Notes |
|----------|---------|----------|----------|------|-------|
| **Pyannote 3.0** | pyannote.audio | SOTA | ⚠️ HF Token | ✅ | Best accuracy, nhưng cần online auth |
| **Resemblyzer** | resemblyzer | Good | ✅ | ❌ | Lightweight, đã implement |
| **NeMo MSDD** | nemo_toolkit | Very Good | ✅ | ✅ | NVIDIA's solution |
| **WhisperX** | whisperx | Good | ⚠️ | ✅ | Word-level alignment + diarization |
| **Simple VAD** | webrtcvad | Poor | ✅ | ❌ | Chỉ phân biệt speech/silence |

### 4.4 Deep Dive: Resemblyzer (Đang Dùng)

#### Cơ Chế
1. **VoiceEncoder**: Pre-trained d-vector model (256-dim embeddings)
2. **Embedding**: Mỗi audio segment → 1 vector đại diện giọng nói
3. **Clustering**: Spectral/K-means để nhóm segments theo speaker

#### Ưu Điểm
- ✅ Offline hoàn toàn (model ~17MB)
- ✅ Không cần GPU
- ✅ Đơn giản, dễ customize

#### Nhược Điểm
- ❌ Accuracy thấp hơn Pyannote
- ❌ Cần biết số speakers trước (`n_speakers` param)
- ❌ Không có VAD tích hợp

### 4.5 Deep Dive: Pyannote 3.0 (Best-in-class)

#### Kiến Trúc
```
Audio → Segmentation Model → Voice Activity Detection
                          → Speaker Change Detection
                          → Overlapped Speech Detection
      → Embedding Model → Speaker Embeddings
      → Clustering → Final Diarization
```

#### Metrics (AMI Dataset)
- DER (Diarization Error Rate): ~5.5%
- Overlap handling: Yes

#### Hạn Chế Cho Cherry Core
- Cần HuggingFace token (online authentication)
- Model size: ~2GB
- **Giải pháp**: Pre-download models, cache locally

### 4.6 Đề Xuất: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: VAD (Silero/WebRTC)                                │
│ - Loại bỏ silence segments                                  │
│ - Tránh Whisper hallucination                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: ASR (Whisper V2)                                   │
│ - Transcribe với word-level timestamps                      │
│ - Áp dụng anti-hallucination configs                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Diarization (Resemblyzer)                          │
│ - Speaker segmentation song song với ASR                    │
│ - Output: [(start, end, speaker_id), ...]                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Alignment                                          │
│ - Map ASR words → Speaker segments                          │
│ - Xử lý overlaps và edge cases                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Speaker-Aware Transcript                           │
│ - Format: "[SPEAKER_1]: text..."                            │
│ - Role inference: Caller/Receiver based on pattern          │
└─────────────────────────────────────────────────────────────┘
```

### 4.7 Integration Code Proposal

```python
# application/services/transcription_service.py (updated)
class TranscriptionService:
    def __init__(self, transcriber: ITranscriber, diarizer: ISpeakerDiarizer = None):
        self.transcriber = transcriber
        self.diarizer = diarizer
    
    def transcribe_with_speakers(self, audio_path: str) -> Transcript:
        # 1. ASR with word timestamps
        transcript = self.transcriber.transcribe(audio_path)
        
        if not self.diarizer:
            return transcript
        
        # 2. Diarization
        speaker_segments = self.diarizer.diarize(audio_path)
        
        # 3. Alignment
        aligned_segments = self._align_speakers_to_words(
            transcript.segments, 
            speaker_segments
        )
        
        # 4. Format output
        transcript.text = self._format_speaker_text(aligned_segments)
        transcript.metadata['speakers'] = self._extract_speaker_info(speaker_segments)
        
        return transcript
    
    def _align_speakers_to_words(self, words, speakers):
        """Align ASR word timestamps with speaker segments."""
        aligned = []
        for word in words:
            word_mid = (word['start'] + word['end']) / 2
            speaker = self._find_speaker_at_time(word_mid, speakers)
            aligned.append({**word, 'speaker': speaker})
        return aligned
```

---

## 5. Đề Xuất Nâng Cấp

### Phase 1: ASR Optimization (Ưu Tiên Cao)
**Mục tiêu**: Giảm 50% lỗi phonetic

| Task | File | Effort |
|------|------|--------|
| Tích hợp Silero VAD preprocessing | `whisperv2_adapter.py` | 2 days |
| Thêm n-gram penalty | `whisperv2_adapter.py` | 1 day |
| Enhanced repetition filter (Vietnamese-specific) | `whisperv2_adapter.py` | 1 day |

### Phase 2: Chain-of-Verification (CoVe)
**Mục tiêu**: Tự động detect và correct LLM factual errors

| Task | File | Effort |
|------|------|--------|
| Create CoVeService skeleton | `application/services/cove_service.py` (new) | 2 days |
| Verification prompts | `prompts/templates/cove_*.j2` (new) | 1 day |
| Integration với AnalysisService | `analysis_service.py` | 1 day |
| Testing & tuning | - | 2 days |

### Phase 3: Speaker Diarization Integration
**Mục tiêu**: Xác định "Ai nói câu nào"

| Task | File | Effort |
|------|------|--------|
| ASR word-level timestamps | `whisperv2_adapter.py` | 1 day |
| Alignment algorithm | `transcription_service.py` | 2 days |
| Speaker-aware prompts | `deep_investigation.j2` | 1 day |
| Add diarizer to SystemFactory | `system_factory.py` | 0.5 day |

### Phase 4: Structured Output Guarantee
**Mục tiêu**: 100% valid JSON output

| Task | File | Effort |
|------|------|--------|
| Evaluate XGrammar vs current GBNF | - | 1 day |
| Pydantic schemas for reports | `core/domain/schemas.py` (new) | 1 day |
| Re-enable grammar with optimization | `llamacpp_adapter.py` | 1 day |

### Phase 5: Vietnamese Correction Enhancement
**Mục tiêu**: Sửa 90%+ lỗi homophone

| Task | File | Effort |
|------|------|--------|
| Expand common_errors dictionary | `vn_custom_vocabulary.json` | 1 day |
| Rule-based pre-processor | `correction_service.py` | 1 day |
| Multi-pass correction pipeline | `transcribe_audio.py` | 1 day |

### Phase 6: RAG Enhancement
**Mục tiêu**: Semantic vocabulary retrieval

| Task | File | Effort |
|------|------|--------|
| Vector store adapter (FAISS) | `infrastructure/adapters/rag/` (new) | 2 days |
| Embedding service | `application/services/embedding_service.py` (new) | 1 day |
| Replace keyword matching | `correction_service.py` | 1 day |

---

## 6. Implementation Roadmap

### Sprint 1 (Tuần 1-2): ASR + Diarization Foundation
- [ ] VAD preprocessing
- [ ] N-gram penalty
- [ ] ASR word-level timestamps
- [ ] Basic diarization integration

### Sprint 2 (Tuần 3-4): CoVe Implementation
- [ ] CoVeService
- [ ] Verification prompts
- [ ] AnalysisService integration
- [ ] Testing với sample transcripts

### Sprint 3 (Tuần 5-6): Structured Output + Correction
- [ ] XGrammar evaluation
- [ ] Pydantic schemas
- [ ] Common errors expansion
- [ ] Multi-pass correction

### Sprint 4 (Tuần 7-8): RAG + Polish
- [ ] Vector store setup
- [ ] Semantic retrieval
- [ ] End-to-end testing
- [ ] Documentation update

---

## 7. Tài Liệu Tham Khảo

### Papers
1. Dhuliawala et al. (2024). "Chain-of-Verification Reduces Hallucination in Large Language Models". ACL Findings.
2. "Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down" (2025). arXiv:2505.12969
3. "VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data" (2025). arXiv:2505.21527
4. "MEGA-RAG: Multi-Evidence Guided Answer Refinement for Mitigating Hallucinations" (2025). PMC
5. "XGrammar: Flexible and Efficient Structured Generation Engine for LLMs" (2024). Preprint.
6. "A Comprehensive Survey of Hallucination in LLMs" (2025). arXiv:2510.06265

### Libraries
- **Resemblyzer**: https://github.com/resemble-ai/Resemblyzer
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **Silero VAD**: https://github.com/snakers4/silero-vad
- **XGrammar**: Integrated in vLLM v1.0+
- **sentence-transformers**: https://www.sbert.net/

### Benchmarks
- TruthfulQA: Hallucination evaluation
- FACTSCORE: Factual precision for biography generation
- DER (Diarization Error Rate): Speaker diarization accuracy

---

## Appendix A: Metrics Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| ASR WER (Vietnamese) | ~15-20% | <10% | Manual annotation sample |
| JSON Parse Success | ~95% | 100% | Log analysis |
| Hallucination Rate | Unknown | <5% | CoVe verification pass rate |
| Correction Accuracy | ~70% | >90% | Before/after comparison |
| Diarization DER | N/A | <15% | Manual annotation |

## Appendix B: Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| XGrammar compatibility với GGUF | High | Fallback to current GBNF |
| Diarization accuracy thấp trên Vietnamese | Medium | Tune clustering params, add heuristics |
| CoVe tăng latency | Medium | Cache verification results, async processing |
| Pyannote requires HF token | Low | Pre-download models, use Resemblyzer as fallback |
