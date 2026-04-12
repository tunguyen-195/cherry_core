# CYCLE 2 RESEARCH: SPEAKER DIARIZATION (OFFLINE)

## 1. OBJECTIVE
Add speaker identification to transcripts ("Ai nói câu nào?") to improve LLM analysis quality.

---

## 2. RESEARCH FINDINGS

### 2.1 Available Approaches

| Approach | Library | Pros | Cons | Offline? |
| :--- | :--- | :--- | :--- | :--- |
| **Pyannote** | `pyannote.audio` | State-of-the-art accuracy | GPU required, ~2GB model, HF Token needed | ⚠️ Partial |
| **Resemblyzer** | `resemblyzer` | Lightweight, fast embeddings | Requires manual clustering | ✅ Yes |
| **Simple Energy VAD** | Custom | No extra deps | Very low accuracy | ✅ Yes |
| **Whisper-X** | `whisper-x` | Word-level alignment + diarization | Heavy, online dependencies | ⚠️ Partial |

### 2.2 Recommendation: Hybrid Approach
Given the **offline constraint**:
1. **Primary**: Use **Resemblyzer** (Speaker Embeddings) + Spectral Clustering.
2. **Fallback**: Heuristic segmentation based on silence gaps.

---

## 3. TECHNICAL DESIGN

### 3.1 New Port (Interface)
```python
# core/ports/diarization_port.py
from abc import ABC, abstractmethod
from typing import List
from core.domain.transcript import SpeakerSegment

class ISpeakerDiarizer(ABC):
    @abstractmethod
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        """Return list of segments with speaker labels."""
        pass
```

### 3.2 Domain Entity
```python
# core/domain/transcript.py (extend)
@dataclass
class SpeakerSegment:
    start_time: float
    end_time: float
    speaker_id: str  # "SPEAKER_1", "SPEAKER_2", etc.
    text: str = ""   # Filled after ASR alignment
```

### 3.3 Adapter Implementation
```python
# infrastructure/adapters/diarization/resemblyzer_adapter.py
class ResemblyzerAdapter(ISpeakerDiarizer):
    def __init__(self):
        from resemblyzer import VoiceEncoder, preprocess_wav
        self.encoder = VoiceEncoder()
    
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        # 1. Load and preprocess audio
        # 2. Split by VAD
        # 3. Compute embeddings for each segment
        # 4. Cluster embeddings (Spectral/KMeans)
        # 5. Return labeled segments
        pass
```

### 3.4 Integration into TranscriptionService
The service will:
1. Run ASR (word-level timestamps if possible).
2. Run Diarization (speaker segments).
3. Merge: Assign speaker IDs to ASR words based on timestamp overlap.

---

## 4. IMPLEMENTATION PLAN

1. **Install**: `pip install resemblyzer scikit-learn`
2. **Create Port**: `core/ports/diarization_port.py`
3. **Create Entity**: Update `core/domain/transcript.py`
4. **Create Adapter**: `infrastructure/adapters/diarization/resemblyzer_adapter.py`
5. **Update Factory**: Add `create_diarizer()` to `SystemFactory`
6. **Integrate**: Modify `TranscriptionService` to merge ASR + Diarization
7. **Verify**: Test on `test_audio.mp3`

---

## 5. DEPENDENCIES
```
resemblyzer>=0.1.3
scikit-learn>=1.0
```

---

## 6. RISKS & MITIGATIONS

| Risk | Mitigation |
| :--- | :--- |
| Resemblyzer model not offline | Pre-download encoder weights (`pretrained.pt`) |
| Low accuracy on Vietnamese | Acceptable for v1; iterate later |
| Increased processing time | Run in parallel with ASR if possible |

---

**Status**: RESEARCH COMPLETE. Ready for Implementation.
