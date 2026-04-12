import importlib
import logging
import re
from pathlib import Path
from typing import Any

import torch

from core.config import MODELS_DIR, WHISPER_V2_PATH
from core.domain.entities import Transcript
from core.ports.asr_port import ITranscriber

logger = logging.getLogger(__name__)


class WhisperV2Adapter(ITranscriber):
    """
    Whisper large-v2 adapter backed by faster-whisper/CTranslate2.

    The public contract intentionally stays the same as the previous
    ``whisper-v2`` adapter, but the runtime no longer depends on
    ``openai-whisper`` and therefore does not inherit the NumPy/numba issue.
    """

    REPETITION_PATTERN = re.compile(r"(\b\w+\b)(\s*[.,]?\s*\1){2,}", re.IGNORECASE)
    VIETNAMESE_REPETITION_PATTERN = re.compile(
        r"(\b[\w\u00C0-\u1EF9]+\b)(\s*[.,]?\s*\1){2,}",
        re.IGNORECASE,
    )
    DEFAULT_VAD_PARAMETERS = {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 120,
    }

    def __init__(
        self,
        use_vad: bool = True,
        device: str | None = None,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
        num_workers: int = 1,
    ):
        self.model = None
        self._fw_module = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.use_vad = use_vad
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.model_path = self.get_local_model_path()

    @classmethod
    def get_local_model_path(cls) -> Path | None:
        for candidate in cls._candidate_model_paths():
            if cls._is_ct2_model_dir(candidate):
                return candidate
        return None

    @classmethod
    def _candidate_model_paths(cls) -> list[Path]:
        candidates: list[Path] = [
            WHISPER_V2_PATH,
            MODELS_DIR / "faster-whisper-large-v2",
        ]

        cache_root = MODELS_DIR / "whisperx" / "asr" / "models--Systran--faster-whisper-large-v2"
        refs_main = cache_root / "refs" / "main"
        if refs_main.exists():
            snapshot_id = refs_main.read_text(encoding="utf-8").strip()
            if snapshot_id:
                candidates.append(cache_root / "snapshots" / snapshot_id)

        snapshots_dir = cache_root / "snapshots"
        if snapshots_dir.exists():
            for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
                candidates.append(snapshot)

        return candidates

    @staticmethod
    def _is_ct2_model_dir(path: Path) -> bool:
        return path.exists() and (path / "config.json").exists() and (path / "model.bin").exists()

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if self.model_path is None:
            checked_paths = ", ".join(str(path) for path in self._candidate_model_paths())
            raise FileNotFoundError(
                "Offline faster-whisper large-v2 model missing. "
                f"Checked: {checked_paths}. Run scripts/setup_whisperx_offline.py or place a CTranslate2 model locally."
            )

        logger.info("Loading Whisper V2 from %s with faster-whisper (%s on %s)...", self.model_path, self.compute_type, self.device)
        try:
            if self._fw_module is None:
                self._fw_module = importlib.import_module("faster_whisper")
            model_kwargs: dict[str, Any] = {
                "device": self.device,
                "compute_type": self.compute_type,
                "local_files_only": True,
            }
            if self.device == "cpu" and self.cpu_threads is not None:
                model_kwargs["cpu_threads"] = self.cpu_threads
                model_kwargs["num_workers"] = self.num_workers
            self.model = self._fw_module.WhisperModel(str(self.model_path), **model_kwargs)
            logger.info("✅ Whisper V2 loaded (faster-whisper/CTranslate2).")
        except Exception as exc:
            logger.error("Failed to load Whisper V2: %s", exc)
            raise

    def _remove_repetitions(self, text: str) -> str:
        def replace_repetition(match: re.Match[str]) -> str:
            return match.group(1)

        cleaned = self.REPETITION_PATTERN.sub(replace_repetition, text)
        cleaned = self.VIETNAMESE_REPETITION_PATTERN.sub(r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned != text:
            logger.info("🔧 Removed %s chars of repetition.", len(text) - len(cleaned))
        return cleaned

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _serialize_words(words: list[Any] | None) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for word in words or []:
            token = re.sub(r"\s+", " ", getattr(word, "word", "") or "").strip()
            if not token:
                continue
            serialized.append(
                {
                    "word": token,
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "probability": float(getattr(word, "probability", 0.0) or 0.0),
                }
            )
        return serialized

    def _build_transcribe_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "language": "vi",
            "task": "transcribe",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.0,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.5,
            "word_timestamps": True,
            "vad_filter": self.use_vad,
            "hallucination_silence_threshold": 1.5,
        }
        if self.use_vad:
            kwargs["vad_parameters"] = dict(self.DEFAULT_VAD_PARAMETERS)
        return kwargs

    def transcribe(self, audio_path: str) -> Transcript:
        self._load_model()
        logger.info("Transcribing %s...", audio_path)

        kwargs = self._build_transcribe_kwargs()
        try:
            raw_segments, info = self.model.transcribe(audio_path, **kwargs)
        except TypeError as exc:
            if "hallucination_silence_threshold" not in str(exc):
                raise
            logger.warning("faster-whisper version does not support hallucination_silence_threshold. Retrying without it.")
            kwargs.pop("hallucination_silence_threshold", None)
            raw_segments, info = self.model.transcribe(audio_path, **kwargs)

        segments: list[dict[str, Any]] = []
        segment_texts: list[str] = []

        for seg in raw_segments:
            seg_text = self._remove_repetitions(self._normalize_text(getattr(seg, "text", "")))
            if not seg_text:
                continue
            segment_texts.append(seg_text)
            segments.append(
                {
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                    "text": seg_text,
                    "words": self._serialize_words(getattr(seg, "words", None)),
                }
            )

        full_text = self._remove_repetitions(" ".join(segment_texts).strip())

        return Transcript(
            text=full_text,
            segments=segments,
            metadata={
                "model": "whisper-v2",
                "backend": "faster-whisper",
                "language": getattr(info, "language", "vi"),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "duration": float(getattr(info, "duration", 0.0) or 0.0),
                "anti_hallucination": True,
                "post_processing": "repetition_filter",
                "vad_filter": self.use_vad,
                "compute_type": self.compute_type,
                "device": self.device,
                "cpu_threads": self.cpu_threads,
                "word_timestamps": True,
                "model_path": str(self.model_path) if self.model_path else None,
            },
        )
