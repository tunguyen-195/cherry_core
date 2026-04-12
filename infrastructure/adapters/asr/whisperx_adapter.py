"""
WhisperX Adapter - End-to-End ASR + Diarization (Fix v2)

Implements ITranscriber and ISpeakerDiarizer interfaces using WhisperX.
Provides word-level timestamps and high-quality speaker diarization.

Features:
- faster-whisper for ASR (Whisper large-v2 compatible)
- wav2vec2 for word-level alignment
- pyannote 3.1 for speaker diarization
- Offline-capable when models are pre-downloaded

Usage:
    adapter = WhisperXAdapter()
    segments = adapter.transcribe_and_diarize("audio.mp3")
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch

from core.ports.asr_port import ITranscriber
from core.ports.diarization_port import ISpeakerDiarizer
from core.domain.entities import Transcript, SpeakerSegment

logger = logging.getLogger(__name__)


class WhisperXAdapter(ITranscriber, ISpeakerDiarizer):
    """
    End-to-end ASR + Diarization using WhisperX.
    """
    
    def __init__(
        self,
        model_size: str = "large-v2",
        language: str = "vi",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float16",
        batch_size: int = 16,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        self.model_size = model_size
        self.language = language
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Lazy-loaded models
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None
        
        logger.info(f"WhisperXAdapter initialized: model={model_size}, device={self.device}")
    
    def _ensure_asr_model(self):
        """Load ASR model if not already loaded."""
        if self._asr_model is None:
            import whisperx
            logger.info(f"Loading WhisperX ASR model: {self.model_size}...")
            # load_model returns a FasterWhisperPipeline object
            self._asr_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            logger.info("✅ ASR model loaded")
    
    def _ensure_align_model(self):
        """Load alignment model if not already loaded."""
        if self._align_model is None:
            try:
                import whisperx
                logger.info(f"Loading alignment model for: {self.language}...")
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device
                )
                logger.info("✅ Alignment model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load alignment model: {e}")
                logger.warning("   -> Proceeding without word-level alignment (using coarse timestamps)")
                self._align_model = "DISABLED"
                self._align_metadata = None
    
    def _ensure_diarize_model(self):
        """Load diarization model if not already loaded."""
        if self._diarize_model is None:
            if not self.hf_token:
                logger.warning("HF_TOKEN missing, verifying cache...")
            
            try:
                from whisperx.diarize import DiarizationPipeline
            except ImportError:
                import whisperx
                DiarizationPipeline = whisperx.DiarizationPipeline  # Fallback

            logger.info("Loading diarization model: pyannote/speaker-diarization-3.1...")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            logger.info("✅ Diarization model loaded")
    
    def transcribe(self, audio_path: str) -> Transcript:
        import whisperx
        
        self._ensure_asr_model()
        logger.info(f"Transcribing: {audio_path}")
        
        # FIX: Call transcribe on the model object directly
        # The model object loaded by whisperx.load_model is a wrapper that has .transcribe()
        result = self._asr_model.transcribe(
            audio_path,
            batch_size=self.batch_size,
            language=self.language
        )
        
        # Align
        self._ensure_align_model()
        if self._align_model != "DISABLED":
            try:
                result = whisperx.align(
                    result["segments"],
                    self._align_model,
                    self._align_metadata,
                    audio_path,
                    self.device
                )
            except Exception as e:
                logger.error(f"⚠️ Alignment execution failed: {e}")
        else:
            logger.info("Skipping alignment (model not available)")
        
        # Convert
        full_text = " ".join(seg.get("text", "") for seg in result.get("segments", []))
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", ""),
                "words": seg.get("words", [])
            })
            
        return Transcript(
            text=full_text.strip(),
            segments=segments,
            metadata={"model": f"whisperx-{self.model_size}", "word_timestamps": True}
        )
    
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        self._ensure_diarize_model()
        logger.info(f"Diarizing: {audio_path}")
        
        diarize_segments = self._diarize_model(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        
        segments = []
        for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start_time=segment.start,
                end_time=segment.end,
                speaker_id=speaker,
                text=""
            ))
        return segments
    
    def transcribe_and_diarize(self, audio_path: str) -> List[SpeakerSegment]:
        import whisperx
        
        # 1. Transcribe
        self._ensure_asr_model()
        logger.info(f"[1/4] Transcribing: {audio_path}")
        
        # FIX: Call transcribe on the model object
        result = self._asr_model.transcribe(
            audio_path,
            batch_size=self.batch_size,
            language=self.language
        )
        
        # 2. Align (word-level timestamps)
        self._ensure_align_model()
        if self._align_model != "DISABLED":
            logger.info("[2/4] Aligning words...")
            try:
                result = whisperx.align(
                    result["segments"],
                    self._align_model,
                    self._align_metadata,
                    audio_path,
                    self.device
                )
            except Exception as e:
                logger.error(f"⚠️ Alignment execution failed: {e}")
        else:
            logger.info("[2/4] Skipping alignment (model not available)")
        
        # 3. Diarize
        self._ensure_diarize_model()
        logger.info("[3/4] Diarizing speakers...")
        diarize_segments = self._diarize_model(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        
        # 4. Assign
        logger.info("[4/4] Assigning speakers to words...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Convert
        segments = []
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            segments.append(SpeakerSegment(
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                speaker_id=speaker,
                text=seg.get("text", "").strip(),
                words=seg.get("words", [])
            ))
            
        logger.info(f"✅ Full pipeline complete: {len(segments)} segments")
        return segments

    def cleanup(self):
        import gc
        self._asr_model = None
        self._align_model = None
        self._diarize_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
