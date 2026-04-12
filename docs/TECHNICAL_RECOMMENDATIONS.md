# ĐỀ XUẤT KỸ THUẬT NÂNG CẤP CHERRY CORE V3

**Tài liệu kỹ thuật chi tiết cho việc nâng cấp hệ thống**

---

## 1. NÂNG CẤP DIARIZATION (PRIORITY: CRITICAL)

### 1.1 Tùy chọn 1: Sử dụng Pyannote 3.1 (Recommended)

#### Yêu cầu:
```bash
pip install pyannote.audio>=3.1.0
```

#### Cấu hình:
```bash
# 1. Đăng ký HuggingFace account
# 2. Accept license tại: https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Tạo token tại: https://huggingface.co/settings/tokens
# 4. Set environment variable:
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

#### Code Implementation:
```python
# infrastructure/adapters/diarization/pyannote_v3_adapter.py

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch

class PyannoteV3Adapter(ISpeakerDiarizer):
    """
    Pyannote 3.1 - SOTA End-to-End Neural Diarization.

    Features:
    - 16ms frame resolution (80x better than SpeechBrain)
    - Native overlap handling
    - Neural speaker change detection
    - ~11% DER on AMI/DIHARD benchmarks
    """

    def __init__(self,
                 hf_token: str = None,
                 num_speakers: int = None,
                 min_speakers: int = None,
                 max_speakers: int = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers or 1
        self.max_speakers = max_speakers or 10
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            # Prefer CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            ).to(device)

    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        self._ensure_pipeline()

        # Run with progress hook for long files
        with ProgressHook() as hook:
            diarization = self._pipeline(
                audio_path,
                hook=hook,
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

        segments = []
        speaker_map = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"SPEAKER_{len(speaker_map) + 1}"

            segments.append(SpeakerSegment(
                start_time=turn.start,
                end_time=turn.end,
                speaker_id=speaker_map[speaker]
            ))

        return segments
```

### 1.2 Tùy chọn 2: Tune SpeechBrain (Offline Mode)

Nếu cần 100% offline, tune lại SpeechBrain parameters:

```python
# core/config.py

class DiarizationConfig:
    ENGINE = "speechbrain"  # Force offline mode

    # TUNED PARAMETERS
    SEGMENT_DURATION = 0.5   # Giảm từ 1.2s (x2.4 resolution increase)
    STEP_DURATION = 0.1      # Giảm từ 0.2s (x2 overlap)

    # Use Agglomerative instead of Spectral for stability
    CLUSTERING_TYPE = "agglomerative"
    LINKAGE = "complete"  # More robust than "average" for speech

    # Auto speaker detection
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 6
    SILHOUETTE_THRESHOLD = 0.3  # Minimum silhouette score to accept K
```

```python
# infrastructure/adapters/diarization/speechbrain_adapter.py

def _cluster_embeddings(self, X: np.ndarray, segments: List[tuple] = None) -> np.ndarray:
    """Improved clustering with better K estimation."""

    # 1. Normalize embeddings
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 2. Auto-detect K using improved silhouette analysis
    if self.n_speakers is None:
        k = self._estimate_k_silhouette(X_norm)
    else:
        k = self.n_speakers

    # 3. Agglomerative Clustering (more stable than Spectral)
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="complete"  # Changed from "average"
    ).fit(X_norm)

    initial_labels = clustering.labels_

    # 4. VBx Refinement with TUNED parameters
    from infrastructure.adapters.diarization.vbx_refiner import VBxRefiner

    vbx = VBxRefiner(
        loop_prob=0.75,  # Increased from 0.45 (less switching)
        min_duration=0.3  # Don't allow segments < 300ms
    )

    return vbx.refine(X_norm, initial_labels, segments)

def _estimate_k_silhouette(self, X: np.ndarray) -> int:
    """Improved K estimation using silhouette analysis."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    min_k = config.DiarizationConfig.MIN_SPEAKERS
    max_k = min(config.DiarizationConfig.MAX_SPEAKERS, len(X) - 1)

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="complete"
        ).fit(X)

        score = silhouette_score(X, clustering.labels_, metric="cosine")

        # Only accept if significantly better
        if score > best_score + 0.05:
            best_score = score
            best_k = k

    logger.info(f"K estimation: {best_k} speakers (silhouette: {best_score:.3f})")
    return best_k
```

### 1.3 Cải thiện VBx Refiner

```python
# infrastructure/adapters/diarization/vbx_refiner.py

class VBxRefiner:
    """
    Improved Viterbi-based Resegmentation.

    Changes from original:
    - Higher loop_prob (0.75 vs 0.45) for smoother output
    - Duration constraint to prevent micro-segments
    - Better emission probability scaling
    """

    def __init__(self, loop_prob: float = 0.75, min_duration: float = 0.3):
        self.loop_prob = loop_prob
        self.min_duration = min_duration

    def refine(self, embeddings: np.ndarray, labels: List[int],
               timestamps: List[Tuple[float, float]]) -> List[int]:

        if len(labels) < 2:
            return labels

        unique_speakers = sorted(list(set(labels)))
        n_speakers = len(unique_speakers)
        n_segments = len(labels)

        if n_speakers < 2:
            return labels

        # 1. Calculate Speaker Centroids
        centroids = self._compute_centroids(embeddings, labels, unique_speakers)

        # 2. Compute Emission Probabilities
        emission_matrix = self._compute_emissions(embeddings, centroids, unique_speakers)

        # 3. Add duration constraint
        duration_weights = self._compute_duration_weights(timestamps)

        # 4. Viterbi with duration-weighted transitions
        refined_labels = self._viterbi_decode(
            emission_matrix,
            unique_speakers,
            duration_weights
        )

        return refined_labels

    def _compute_duration_weights(self, timestamps: List[Tuple[float, float]]) -> np.ndarray:
        """Penalize very short segments."""
        weights = []
        for start, end in timestamps:
            duration = end - start
            if duration < self.min_duration:
                # Penalize short segments (encourage merging)
                weights.append(0.5)
            else:
                weights.append(1.0)
        return np.array(weights)
```

---

## 2. NÂNG CẤP ASR (PRIORITY: HIGH)

### 2.1 Chuyển sang faster-whisper

```python
# infrastructure/adapters/asr/faster_whisper_adapter.py

from faster_whisper import WhisperModel
from typing import List, Dict

class FasterWhisperAdapter(ITranscriber):
    """
    faster-whisper implementation với:
    - 4x faster inference (CTranslate2 backend)
    - Built-in VAD filter
    - Word-level timestamps
    """

    def __init__(self,
                 model_size: str = "large-v3",
                 device: str = "auto",
                 compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,  # Offline mode
                download_root=str(config.MODELS_DIR)
            )

    def transcribe(self, audio_path: str) -> Transcript:
        self._ensure_model()

        segments, info = self._model.transcribe(
            audio_path,
            language="vi",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,  # Built-in VAD
            vad_parameters=dict(
                min_speech_duration_ms=100,
                min_silence_duration_ms=300,
                speech_pad_ms=200
            ),
            word_timestamps=True,  # Enable word-level
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.5
        )

        # Process segments with word timestamps
        processed_segments = []
        all_words = []
        full_text = []

        for segment in segments:
            seg_text = self._clean_text(segment.text)
            if not seg_text:
                continue

            full_text.append(seg_text)

            processed_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": seg_text
            })

            # Collect word-level timestamps
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.probability
                    })

        return Transcript(
            text=" ".join(full_text),
            segments=processed_segments,
            metadata={
                "model": f"faster-whisper-{self.model_size}",
                "language": info.language,
                "duration": info.duration,
                "words": all_words  # Word-level for diarization alignment
            }
        )

    def _clean_text(self, text: str) -> str:
        """Advanced Vietnamese repetition cleaning."""
        import re

        text = text.strip()
        if not text:
            return ""

        # Pattern 1: Word repetition (3+)
        text = re.sub(
            r'(\b[\w\u00C0-\u1EF9]+\b)(\s*[.,!?]?\s*\1){2,}',
            r'\1',
            text,
            flags=re.IGNORECASE
        )

        # Pattern 2: Phrase repetition (2+)
        text = re.sub(
            r'([\w\u00C0-\u1EF9\s]{4,20})(\s*[.,!?]?\s*\1){1,}',
            r'\1',
            text,
            flags=re.IGNORECASE
        )

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
```

### 2.2 Word-to-Speaker Alignment

```python
# infrastructure/adapters/alignment/word_speaker_aligner.py

from typing import List, Dict
from core.domain.entities import SpeakerSegment, Transcript

class WordSpeakerAligner:
    """
    Align word-level ASR output with speaker diarization segments.

    Algorithm:
    1. For each word, find overlapping speaker segment
    2. Assign word to speaker with maximum overlap
    3. Group consecutive words by speaker into sentences
    """

    def align(self,
              words: List[Dict],  # From ASR: [{word, start, end}, ...]
              speaker_segments: List[SpeakerSegment]
              ) -> List[SpeakerSegment]:
        """
        Align words to speakers.

        Args:
            words: Word-level transcription with timestamps
            speaker_segments: Speaker diarization output

        Returns:
            List of SpeakerSegment with text filled in
        """
        if not words or not speaker_segments:
            return speaker_segments

        # Build speaker interval tree for fast lookup
        from intervaltree import IntervalTree
        speaker_tree = IntervalTree()

        for seg in speaker_segments:
            speaker_tree[seg.start_time:seg.end_time] = seg.speaker_id

        # Assign each word to a speaker
        word_speakers = []
        for word in words:
            word_mid = (word["start"] + word["end"]) / 2

            # Find overlapping speaker
            overlaps = speaker_tree[word_mid]

            if overlaps:
                # Take the first match (should usually be one)
                speaker = list(overlaps)[0].data
            else:
                # Find nearest speaker if no overlap
                speaker = self._find_nearest_speaker(word_mid, speaker_segments)

            word_speakers.append({
                **word,
                "speaker": speaker
            })

        # Group consecutive words by speaker
        return self._group_by_speaker(word_speakers)

    def _find_nearest_speaker(self, time: float, segments: List[SpeakerSegment]) -> str:
        """Find nearest speaker when no overlap exists."""
        min_dist = float("inf")
        nearest = segments[0].speaker_id

        for seg in segments:
            dist = min(abs(seg.start_time - time), abs(seg.end_time - time))
            if dist < min_dist:
                min_dist = dist
                nearest = seg.speaker_id

        return nearest

    def _group_by_speaker(self, word_speakers: List[Dict]) -> List[SpeakerSegment]:
        """Group consecutive words by speaker into segments."""
        if not word_speakers:
            return []

        segments = []
        current_speaker = word_speakers[0]["speaker"]
        current_words = [word_speakers[0]]

        for word in word_speakers[1:]:
            if word["speaker"] == current_speaker:
                current_words.append(word)
            else:
                # Finalize current segment
                segments.append(SpeakerSegment(
                    start_time=current_words[0]["start"],
                    end_time=current_words[-1]["end"],
                    speaker_id=current_speaker,
                    text=" ".join(w["word"] for w in current_words),
                    words=current_words
                ))

                # Start new segment
                current_speaker = word["speaker"]
                current_words = [word]

        # Don't forget last segment
        if current_words:
            segments.append(SpeakerSegment(
                start_time=current_words[0]["start"],
                end_time=current_words[-1]["end"],
                speaker_id=current_speaker,
                text=" ".join(w["word"] for w in current_words),
                words=current_words
            ))

        return segments
```

---

## 3. VIETNAMESE POST-PROCESSING (PRIORITY: MEDIUM)

### 3.1 Domain-Specific Corrections

```python
# infrastructure/adapters/correction/vietnamese_postprocessor.py

class VietnamesePostProcessor:
    """
    Vietnamese-specific post-processing for ASR output.

    Features:
    - Domain-specific term correction
    - Number formatting
    - PII detection and formatting
    - Common ASR error patterns
    """

    # Hotel domain corrections
    HOTEL_CORRECTIONS = {
        "đi lặn": "Deluxe",
        "đi lắc": "Deluxe",
        "vòng x kế tiếp": "Executive",
        "x kế tiếp": "Executive",
        "điều trú": "lưu trú",
        "cộng phận": "bộ phận",
        "căn cứ công dân": "căn cước công dân",
        "căn cướp": "căn cước",
        "quỷ trả": "hủy trả",
        "nghiêm tiếp": "niêm yết",
        "giá nhịp ít": "giá niêm yết",
        "Xin ký": "Xin kính",
        "cái sạn": "khách sạn",
        "đi trú": "lưu trú",
    }

    # Common Vietnamese ASR errors
    COMMON_CORRECTIONS = {
        "dạ vâng ạ": "Dạ vâng ạ",
        "ờ": "Ờ",
        "à": "À",
    }

    # Number patterns
    PHONE_PATTERN = r'\b(\d{4})\s*(\d{3})\s*(\d{3})\b'
    PHONE_REPLACEMENT = r'\1 \2 \3'

    def process(self, text: str, domain: str = "hotel") -> str:
        """Apply all post-processing steps."""

        # 1. Domain-specific corrections
        if domain == "hotel":
            text = self._apply_corrections(text, self.HOTEL_CORRECTIONS)

        # 2. Common corrections
        text = self._apply_corrections(text, self.COMMON_CORRECTIONS)

        # 3. Format phone numbers
        text = self._format_phone_numbers(text)

        # 4. Format currency
        text = self._format_currency(text)

        # 5. Capitalize proper nouns
        text = self._capitalize_proper_nouns(text)

        return text

    def _apply_corrections(self, text: str, corrections: Dict[str, str]) -> str:
        for wrong, correct in corrections.items():
            text = re.sub(
                rf'\b{re.escape(wrong)}\b',
                correct,
                text,
                flags=re.IGNORECASE
            )
        return text

    def _format_phone_numbers(self, text: str) -> str:
        # Format: 0978 711 253
        import re
        return re.sub(self.PHONE_PATTERN, self.PHONE_REPLACEMENT, text)

    def _format_currency(self, text: str) -> str:
        # Format: 3 triệu -> 3,000,000 VND (optional)
        # Keep original for now
        return text

    def _capitalize_proper_nouns(self, text: str) -> str:
        proper_nouns = [
            "JW Marriott", "Marriott", "Fitness Center",
            "Hà Nội", "Việt Nam"
        ]
        for noun in proper_nouns:
            text = re.sub(
                rf'\b{re.escape(noun)}\b',
                noun,
                text,
                flags=re.IGNORECASE
            )
        return text
```

---

## 4. FULL PIPELINE INTEGRATION

### 4.1 Integrated Transcription Service

```python
# application/services/integrated_transcriber.py

class IntegratedTranscriptionService:
    """
    Full pipeline: Audio -> Diarized Transcript

    Pipeline:
    1. VAD Preprocessing
    2. ASR with word timestamps
    3. Speaker Diarization
    4. Word-Speaker Alignment
    5. Post-processing
    """

    def __init__(self,
                 asr_adapter: ITranscriber,
                 diarizer: ISpeakerDiarizer,
                 postprocessor: VietnamesePostProcessor = None):
        self.asr = asr_adapter
        self.diarizer = diarizer
        self.aligner = WordSpeakerAligner()
        self.postprocessor = postprocessor or VietnamesePostProcessor()

    def transcribe_with_speakers(self,
                                  audio_path: str,
                                  domain: str = "general"
                                  ) -> List[SpeakerSegment]:
        """
        Transcribe audio with speaker diarization.

        Returns:
            List of SpeakerSegment with:
            - start_time, end_time
            - speaker_id
            - text (post-processed)
            - words (word-level details)
        """

        # 1. ASR with word timestamps
        transcript = self.asr.transcribe(audio_path)
        words = transcript.metadata.get("words", [])

        if not words:
            logger.warning("No word timestamps available. Using segment-level alignment.")
            return self._segment_level_fallback(transcript, audio_path)

        # 2. Speaker Diarization
        speaker_segments = self.diarizer.diarize(audio_path)

        # 3. Word-Speaker Alignment
        aligned_segments = self.aligner.align(words, speaker_segments)

        # 4. Post-processing
        for segment in aligned_segments:
            segment.text = self.postprocessor.process(segment.text, domain=domain)

        return aligned_segments

    def _segment_level_fallback(self, transcript: Transcript, audio_path: str) -> List[SpeakerSegment]:
        """Fallback when word timestamps are not available."""

        speaker_segments = self.diarizer.diarize(audio_path)

        # Simple text distribution based on duration ratio
        total_duration = sum(s.end_time - s.start_time for s in speaker_segments)
        words = transcript.text.split()
        words_per_second = len(words) / total_duration if total_duration > 0 else 1

        word_idx = 0
        for segment in speaker_segments:
            duration = segment.end_time - segment.start_time
            n_words = int(duration * words_per_second)

            segment_words = words[word_idx:word_idx + n_words]
            segment.text = self.postprocessor.process(" ".join(segment_words))

            word_idx += n_words

        return speaker_segments
```

### 4.2 Output Formatter

```python
# application/services/output_formatter.py

class OutputFormatter:
    """Format diarized output in various formats."""

    @staticmethod
    def to_txt(segments: List[SpeakerSegment], include_timestamps: bool = True) -> str:
        """Format as plain text."""
        lines = []

        for seg in segments:
            if include_timestamps:
                start = OutputFormatter._format_time(seg.start_time)
                end = OutputFormatter._format_time(seg.end_time)
                lines.append(f"{start} --> {end} [{seg.speaker_id}]")
            else:
                lines.append(f"[{seg.speaker_id}]")

            lines.append(seg.text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_srt(segments: List[SpeakerSegment]) -> str:
        """Format as SRT subtitle."""
        lines = []

        for i, seg in enumerate(segments, 1):
            start = OutputFormatter._format_time_srt(seg.start_time)
            end = OutputFormatter._format_time_srt(seg.end_time)

            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"[{seg.speaker_id}] {seg.text}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(segments: List[SpeakerSegment]) -> dict:
        """Format as JSON."""
        return {
            "segments": [
                {
                    "start": seg.start_time,
                    "end": seg.end_time,
                    "speaker": seg.speaker_id,
                    "text": seg.text,
                    "words": seg.words if seg.words else None
                }
                for seg in segments
            ]
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time as HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Format time for SRT (same as above but with comma)."""
        return OutputFormatter._format_time(seconds)
```

---

## 5. REQUIREMENTS.TXT ĐỀ XUẤT

```txt
# Core Dependencies
torch>=2.0.0
transformers>=4.36.0
numpy>=1.24.0
scipy>=1.11.0
soundfile>=0.12.0
librosa>=0.10.0

# ASR
faster-whisper>=1.0.0
openai-whisper>=20231117

# Speaker Diarization
pyannote.audio>=3.1.0
speechbrain>=1.0.0

# VAD
silero-vad>=4.0.0

# Clustering
scikit-learn>=1.3.0

# LLM
llama-cpp-python>=0.2.0

# Utilities
intervaltree>=3.1.0
pyyaml>=6.0
jinja2>=3.1.0
tqdm>=4.66.0

# Optional: GPU acceleration
# cupy-cuda12x>=12.0.0
```

---

## 6. TESTING RECOMMENDATIONS

### 6.1 Unit Tests

```python
# tests/test_diarization.py

import pytest
from infrastructure.adapters.diarization.speechbrain_adapter import SpeechBrainAdapter

class TestDiarization:

    def test_speaker_count_estimation(self):
        """Test that K estimation works correctly."""
        adapter = SpeechBrainAdapter(n_speakers=None)
        # Should auto-detect 2 speakers for test audio
        segments = adapter.diarize("samples/test_audio.mp3")

        speakers = set(s.speaker_id for s in segments)
        assert len(speakers) == 2

    def test_temporal_resolution(self):
        """Test that segments are not too long."""
        adapter = SpeechBrainAdapter(n_speakers=2)
        segments = adapter.diarize("samples/test_audio.mp3")

        # No segment should be longer than 30 seconds
        for seg in segments:
            duration = seg.end_time - seg.start_time
            assert duration <= 30.0

    def test_vbx_refinement(self):
        """Test VBx doesn't over-smooth."""
        # Check that rapid speaker changes are preserved
        pass
```

### 6.2 Integration Tests

```python
# tests/test_full_pipeline.py

def test_end_to_end_transcription():
    """Test full pipeline from audio to diarized transcript."""

    service = IntegratedTranscriptionService(
        asr_adapter=FasterWhisperAdapter(),
        diarizer=PyannoteV3Adapter(),
        postprocessor=VietnamesePostProcessor()
    )

    segments = service.transcribe_with_speakers(
        "samples/test_audio.mp3",
        domain="hotel"
    )

    # Verify output
    assert len(segments) > 0
    assert all(s.text.strip() for s in segments)
    assert all(s.speaker_id.startswith("SPEAKER_") for s in segments)
```

---

## 7. DEPLOYMENT CHECKLIST

### Pre-deployment:
- [ ] Download all models to `models/` directory
- [ ] Test offline operation
- [ ] Verify HF_TOKEN if using Pyannote
- [ ] Run full test suite
- [ ] Benchmark performance on sample audio

### Model Downloads:
```bash
# Run once to download all models
python scripts/setup_models.py --all
```

### Environment Variables:
```bash
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES="0"  # If using GPU
```

---

*Tài liệu kỹ thuật được tạo bởi AI Research Assistant*
