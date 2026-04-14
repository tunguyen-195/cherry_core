"""
PhoWhisper ASR Adapter (VinAI - Vietnamese SOTA).
Uses locally downloaded model for OFFLINE operation.

Benchmark WER:
- VIVOS: 4.67% (vs Whisper V2 ~10-15%)
- CMV-Vi: 8.14%

Source: https://github.com/VinAIResearch/PhoWhisper
"""
import copy
import logging
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

from core.ports.asr_port import ITranscriber
from core.domain.entities import Transcript
from core.config import MODELS_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)


class PhoWhisperAdapter(ITranscriber):
    """
    Vietnamese ASR using PhoWhisper (VinAI Research).
    SOTA for Vietnamese with 4.67% WER on VIVOS.
    
    Uses local model for OFFLINE operation.
    Returns word-level timestamps for optimal diarization alignment.
    """
    
    # Local model paths (already downloaded)
    # IMPORTANT: phowhisper-safe with safetensors format is prioritized
    # to avoid CVE-2025-32434 vulnerability with torch 2.5 + pytorch_model.bin
    MODEL_PATHS = [
        MODELS_DIR / "phowhisper-safe",   # SafeTensors format (CVE-safe) - PRIORITY
        MODELS_DIR / "phowhisper",        # PyTorch format (backup)
        MODELS_DIR / "phowhisper-full",   # Alternate
    ]
    
    def __init__(self, device: str = None):
        """
        Args:
            device: "cpu" or "cuda" (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._model_path = None
        self._forced_decoder_ids = None
        self._generation_config = None
        
    def _find_model_path(self) -> Path:
        """Find available local model path, prioritizing safetensors format."""
        
        # First priority: safetensors format (CVE-2025-32434 safe)
        for path in self.MODEL_PATHS:
            if path.exists():
                # Check for sharded safetensors (model-00001-of-*.safetensors)
                safetensors_files = list(path.glob("model-*.safetensors"))
                if safetensors_files:
                    return path
                # Check for single safetensors
                if (path / "model.safetensors").exists():
                    return path
        
        # Second priority: pytorch_model.bin (may fail with CVE check)
        for path in self.MODEL_PATHS:
            if path.exists() and (path / "pytorch_model.bin").exists():
                return path
        
        # Fallback: any folder with config
        for path in self.MODEL_PATHS:
            if path.exists() and (path / "config.json").exists():
                return path
        
        raise FileNotFoundError(
            f"PhoWhisper model not found. Expected at: {self.MODEL_PATHS[0]}\n"
            "Run: python scripts/setup_models.py"
        )
    
    def _load_model(self):
        """Load model from local path."""
        if self._model is not None:
            return
        
        self._model_path = self._find_model_path()
        logger.info(f"🇻🇳 Loading PhoWhisper from: {self._model_path}")
        
        try:
            self._processor = WhisperProcessor.from_pretrained(
                str(self._model_path),
                local_files_only=True
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                str(self._model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True,
                use_safetensors=True
            )
            self._model.to(self.device)
            self._model.eval()
            tokenizer = getattr(self._processor, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "get_decoder_prompt_ids"):
                raise RuntimeError("PhoWhisper tokenizer does not support decoder prompt ids.")
            self._forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="vi", task="transcribe")
            base_generation_config = getattr(self._model, "generation_config", None)
            if base_generation_config is not None:
                self._generation_config = copy.deepcopy(base_generation_config)
                self._generation_config.forced_decoder_ids = self._forced_decoder_ids
            
            logger.info(f"✅ PhoWhisper loaded on {self.device.upper()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PhoWhisper: {e}")
            raise

    @classmethod
    def runtime_ready(cls) -> bool:
        try:
            import soundfile  # noqa: F401
            import transformers  # noqa: F401
            import torchaudio  # noqa: F401
        except Exception:
            return False

        try:
            model_path = cls(device="cpu")._find_model_path()
        except FileNotFoundError:
            return False

        required_files = [
            model_path / "config.json",
            model_path / "preprocessor_config.json",
            model_path / "tokenizer.json",
        ]
        has_safe_weights = bool(list(model_path.glob("model-*.safetensors"))) or (model_path / "model.safetensors").exists()
        has_bin_weights = (model_path / "pytorch_model.bin").exists()
        return all(path.exists() for path in required_files) and (has_safe_weights or has_bin_weights)

    def transcribe(self, audio_path: str) -> Transcript:
        """
        Transcribe audio with word-level timestamps.
        
        Returns Transcript with segments containing 'words' list
        for optimal diarization alignment.
        """
        self._load_model()
        
        logger.info(f"🎙️ PhoWhisper transcribing: {audio_path}")
        
        # Load and preprocess audio
        waveform, _sample_rate = self._load_audio(audio_path)
        
        # Get audio duration
        audio_duration = waveform.shape[1] / SAMPLE_RATE
        
        # Process in chunks for long audio (Whisper limit: 30s)
        chunk_length = 30.0  # seconds
        chunk_results = []
        
        if audio_duration <= chunk_length:
            # Short audio: process in one go
            chunk_results.extend(self._transcribe_chunk(waveform.squeeze().numpy(), 0))
        else:
            # Long audio: process in chunks with overlap
            chunk_samples = int(chunk_length * SAMPLE_RATE)
            step_samples = int(25.0 * SAMPLE_RATE)  # 25s step, 5s overlap
            
            waveform_flat = waveform.squeeze()
            offset = 0
            
            while offset < waveform_flat.shape[0]:
                chunk = waveform_flat[offset:offset + chunk_samples]
                time_offset = offset / SAMPLE_RATE
                
                chunk_results.extend(self._transcribe_chunk(chunk.numpy(), time_offset))
                
                offset += step_samples

        all_segments = self._merge_overlapping_segments(chunk_results)
        full_text = " ".join(segment["text"] for segment in all_segments if segment.get("text")).strip()
        
        logger.info(f"✅ PhoWhisper: {len(all_segments)} segments, {len(full_text)} chars")
        
        return Transcript(
            text=full_text,
            segments=all_segments,
            metadata={
                "model": "PhoWhisper-large",
                "language": "vi",
                "model_path": str(self._model_path),
                "wer_benchmark": "4.67% (VIVOS)"
            }
        )

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """
        Load audio without relying on torchaudio backend codecs.

        torchaudio.load now routes through torchcodec on some builds, which breaks
        clean offline installs on Windows. We keep the repo portable by decoding
        with soundfile and only using torchaudio for in-memory resampling.
        """
        waveform, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        waveform_tensor = torch.from_numpy(np.ascontiguousarray(waveform.T))

        if waveform_tensor.shape[0] > 1:
            waveform_tensor = waveform_tensor.mean(dim=0, keepdim=True)

        if sample_rate != SAMPLE_RATE:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE

        return waveform_tensor.contiguous(), sample_rate
    
    def _transcribe_chunk(self, audio_array, time_offset: float):
        """Transcribe a single audio chunk."""
        
        # Prepare input
        batch = self._processor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_features = batch.input_features.to(self.device)
        attention_mask = getattr(batch, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        if self.device == "cuda":
            input_features = input_features.half()
        
        # Generate (simplified - no timestamps to avoid compatibility issues)
        generation_kwargs = {
            "max_new_tokens": 440,
        }
        if self._generation_config is not None:
            generation_kwargs["generation_config"] = self._generation_config
        elif self._forced_decoder_ids is not None:
            generation_kwargs["forced_decoder_ids"] = self._forced_decoder_ids
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            generated_ids = self._model.generate(input_features, **generation_kwargs)
        
        # Decode
        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        # Create simple segment (we'll use diarization for timing)
        chunk_duration = len(audio_array) / SAMPLE_RATE
        return [{
            "start": time_offset,
            "end": time_offset + chunk_duration,
            "text": text,
            "words": self._estimate_word_timestamps(text, time_offset, time_offset + chunk_duration)
        }] if text else []
    
    def _parse_timestamps(self, text: str, time_offset: float):
        """Parse Whisper timestamp format into segments."""
        import re
        
        segments = []
        pattern = r'<\|([\d.]+)\|>([^<]*)'
        matches = list(re.finditer(pattern, text))
        
        for i, match in enumerate(matches):
            start_time = float(match.group(1)) + time_offset
            segment_text = match.group(2).strip()
            
            if not segment_text:
                continue
            
            # Estimate end time from next timestamp or segment length
            if i + 1 < len(matches):
                end_time = float(matches[i + 1].group(1)) + time_offset
            else:
                # Estimate from text length (~150 words/min for Vietnamese)
                word_count = len(segment_text.split())
                end_time = start_time + max(0.5, word_count * 0.4)
            
            # Create word-level timestamps (estimated)
            words = self._estimate_word_timestamps(segment_text, start_time, end_time)
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "text": segment_text,
                "words": words
            })
        
        return segments
    
    def _estimate_word_timestamps(self, text: str, start: float, end: float):
        """Estimate word-level timestamps based on segment timing."""
        words_list = text.split()
        if not words_list:
            return []
        
        duration = end - start
        word_duration = duration / len(words_list)
        
        words = []
        current_time = start
        
        for word in words_list:
            words.append({
                "word": word,
                "start": round(current_time, 3),
                "end": round(current_time + word_duration, 3)
            })
            current_time += word_duration
        
        return words

    @staticmethod
    def _normalize_word(word: str) -> str:
        return re.sub(r"[^\w]", "", word.lower(), flags=re.UNICODE)

    def _find_chunk_word_overlap(
        self,
        previous_words: list[str],
        current_words: list[str],
        *,
        min_overlap_words: int = 8,
        max_overlap_words: int = 40,
    ) -> int:
        if not previous_words or not current_words:
            return 0

        previous_tail = previous_words[-max_overlap_words:]
        current_head = current_words[:max_overlap_words]
        previous_normalized = [self._normalize_word(word) for word in previous_tail]
        current_normalized = [self._normalize_word(word) for word in current_head]

        max_overlap = min(len(previous_normalized), len(current_normalized))
        for overlap_size in range(max_overlap, min_overlap_words - 1, -1):
            if previous_normalized[-overlap_size:] == current_normalized[:overlap_size]:
                return overlap_size

        for overlap_size in range(max_overlap, min_overlap_words - 1, -1):
            previous_overlap = previous_normalized[-overlap_size:]
            current_overlap = current_normalized[:overlap_size]
            distance = self._token_levenshtein_distance(previous_overlap, current_overlap)
            allowed_distance = max(1, overlap_size // 6)
            if distance <= allowed_distance:
                return overlap_size

        return 0

    @staticmethod
    def _token_levenshtein_distance(source: list[str], target: list[str]) -> int:
        if len(source) < len(target):
            source, target = target, source

        previous = list(range(len(target) + 1))
        for row_index, source_item in enumerate(source, start=1):
            current = [row_index]
            for col_index, target_item in enumerate(target, start=1):
                insert_cost = current[col_index - 1] + 1
                delete_cost = previous[col_index] + 1
                replace_cost = previous[col_index - 1] + (source_item != target_item)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def _merge_overlapping_segments(self, chunk_segments: list[dict]) -> list[dict]:
        merged_segments: list[dict] = []
        merged_words: list[str] = []

        for segment in chunk_segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue

            words = text.split()
            overlap_words = self._find_chunk_word_overlap(merged_words, words)
            if overlap_words:
                words = words[overlap_words:]

            if not words:
                continue

            total_words = max(1, len(text.split()))
            word_ratio = overlap_words / total_words
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            adjusted_start = min(end, start + ((end - start) * word_ratio))
            merged_text = " ".join(words).strip()

            merged_segment = {
                "start": adjusted_start,
                "end": end,
                "text": merged_text,
                "words": self._estimate_word_timestamps(merged_text, adjusted_start, end),
            }
            merged_segments.append(merged_segment)
            merged_words.extend(words)

        return merged_segments
    
    def load(self):
        """Pre-load model for faster first transcription."""
        self._load_model()

