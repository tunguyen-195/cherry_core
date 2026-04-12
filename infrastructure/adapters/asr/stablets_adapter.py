from __future__ import annotations

import logging
from typing import Any

from core.domain.entities import Transcript
from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter
from infrastructure.utils.vendor_imports import import_vendor_module

logger = logging.getLogger(__name__)


class StableTsAdapter(WhisperV2Adapter):
    """
    Offline Stable-TS refinement backed by faster-whisper large-v2.

    This adapter intentionally uses the same local CTranslate2 model as
    ``WhisperV2Adapter`` so it stays fully offline and NumPy-2.4-compatible.
    """

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if self.model_path is None:
            checked_paths = ", ".join(str(path) for path in self._candidate_model_paths())
            raise FileNotFoundError(
                "Offline faster-whisper large-v2 model missing for Stable-TS. "
                f"Checked: {checked_paths}."
            )

        stable_whisper = import_vendor_module("stable_whisper")
        logger.info("Loading Stable-TS from %s (%s on %s)...", self.model_path, self.compute_type, self.device)
        self.model = stable_whisper.load_faster_whisper(
            str(self.model_path),
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,
            cpu_threads=self.cpu_threads if self.device == "cpu" else None,
            num_workers=self.num_workers if self.device == "cpu" else 1,
        )
        logger.info("✅ Stable-TS loaded (offline faster-whisper).")

    @staticmethod
    def runtime_ready() -> bool:
        try:
            import_vendor_module("stable_whisper")
            return WhisperV2Adapter.get_local_model_path() is not None
        except Exception:
            return False

    def transcribe(self, audio_path: str) -> Transcript:
        self._load_model()
        logger.info("Stable-TS transcribing %s...", audio_path)

        result = self.model.transcribe(
            audio_path,
            language="vi",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.5,
            suppress_silence=True,
            suppress_word_ts=True,
            vad=False,
            verbose=None,
        )

        segments: list[dict[str, Any]] = []
        segment_texts: list[str] = []

        for segment in getattr(result, "segments", []) or []:
            seg_text = self._remove_repetitions(self._normalize_text(getattr(segment, "text", "")))
            if not seg_text:
                continue

            segment_texts.append(seg_text)
            words = []
            for word in getattr(segment, "words", []) or []:
                token = self._normalize_text(getattr(word, "word", ""))
                if not token:
                    continue
                words.append(
                    {
                        "word": token,
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "probability": float(getattr(word, "probability", 0.0) or 0.0),
                    }
                )

            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0) or 0.0),
                    "end": float(getattr(segment, "end", 0.0) or 0.0),
                    "text": seg_text,
                    "words": words,
                }
            )

        full_text = self._remove_repetitions(" ".join(segment_texts).strip())
        return Transcript(
            text=full_text,
            segments=segments,
            metadata={
                "model": "stable-ts",
                "backend": "stable-ts+faster-whisper",
                "language": "vi",
                "anti_hallucination": True,
                "timestamp_stabilization": True,
                "suppress_silence": True,
                "word_timestamps": True,
                "compute_type": self.compute_type,
                "device": self.device,
                "cpu_threads": self.cpu_threads,
                "model_path": str(self.model_path) if self.model_path else None,
            },
        )
