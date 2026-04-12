from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from core import config
from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter
from infrastructure.adapters.asr.stablets_adapter import StableTsAdapter
from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter
from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter


class ModelInventoryService:
    """Read-only inventory for local/offline model availability."""

    def get_inventory(self) -> dict[str, Any]:
        phowhisper_path = self._phowhisper_path()
        phowhisper_ready = PhoWhisperAdapter.runtime_ready()
        whisper_v2_path = WhisperV2Adapter.get_local_model_path()
        whisper_v2_ready = whisper_v2_path is not None and self._whisper_v2_runtime_ready()
        vistral_ready = LlamaCppAdapter.runtime_ready("vistral")

        items = [
            self._item(
                model_id="phowhisper",
                family="asr",
                path=phowhisper_path or config.PHOWHISPER_PATH,
                offline_ready=phowhisper_ready,
                notes="Engine ASR mặc định cho tiếng Việt.",
            ),
            self._item(
                model_id="whisper-v2",
                family="asr",
                path=whisper_v2_path or config.WHISPER_V2_PATH,
                offline_ready=whisper_v2_ready,
                notes="Engine dự phòng ổn định, chạy trên faster-whisper/CTranslate2 với cấu hình giảm ảo giác cơ bản.",
            ),
            self._item(
                model_id="whisperx",
                family="asr",
                path=config.MODELS_DIR / "whisperx",
                notes="Backend ASR và alignment offline ở mức thử nghiệm.",
            ),
            self._item(
                model_id="stable-ts",
                family="refinement",
                path=config.BASE_DIR / ".vendor" / "stable_whisper",
                offline_ready=StableTsAdapter.runtime_ready(),
                notes="Lớp ổn định transcript và timestamp chạy offline trên nền faster-whisper large-v2.",
            ),
            self._item(
                model_id="speechbrain",
                family="diarization",
                path=config.SPEECHBRAIN_PATH,
                notes="Backend tách người nói mặc định cho chế độ CPU và offline.",
            ),
            self._item(
                model_id="silero-vad",
                family="vad",
                path=config.SILERO_PATH,
                offline_ready=(config.SILERO_PATH / "silero_vad.jit").exists()
                and (config.SILERO_PATH / "utils_vad.py").exists(),
                notes="Lớp tiền xử lý để giữ lại phần có tiếng nói.",
            ),
            self._item(
                model_id="protonx",
                family="correction",
                path=config.PROTONX_PATH,
                notes="Lớp hiệu chỉnh tiếng Việt dạng seq2seq chạy tùy chọn.",
            ),
            self._item(
                model_id="vistral",
                family="llm",
                path=config.MODELS_DIR / "vistral" / "vistral-7b-chat-Q4_K_M.gguf",
                offline_ready=vistral_ready,
                notes="LLM local cho hiệu chỉnh ngữ cảnh, suy luận vai trò người nói và tóm tắt trinh sát.",
            ),
        ]

        capabilities = {
            "asr_engines": {
                "phowhisper": phowhisper_ready,
                "whisper-v2": whisper_v2_ready,
                "whisperx": self._ready(config.MODELS_DIR / "whisperx"),
            },
            "speaker_modes": {
                "off": True,
                "speechbrain": self._ready(config.SPEECHBRAIN_PATH),
            },
            "features": {
                "apply_vad": self._ready(config.SILERO_PATH / "silero_vad.jit")
                and self._ready(config.SILERO_PATH / "utils_vad.py"),
                "apply_stable_ts": StableTsAdapter.runtime_ready(),
                "apply_protonx": self._ready(config.PROTONX_PATH),
                "apply_llm_correction": vistral_ready,
                "apply_intel_summary": vistral_ready,
                "speaker_refine": vistral_ready,
            },
            "devices": {
                "cpu": True,
                "cuda": torch.cuda.is_available(),
            },
        }

        return {"items": items, "capabilities": capabilities}

    @staticmethod
    def _phowhisper_path() -> Path | None:
        try:
            return PhoWhisperAdapter(device="cpu")._find_model_path()
        except FileNotFoundError:
            return None

    def _item(
        self,
        model_id: str,
        family: str,
        path: Path,
        offline_ready: bool | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        resolved = path if path.is_absolute() else config.BASE_DIR / path
        ready = self._ready(resolved) if offline_ready is None else offline_ready
        return {
            "model_id": model_id,
            "family": family,
            "path": str(resolved),
            "offline_ready": ready,
            "load_status": "unloaded" if ready else "missing",
            "notes": notes,
        }

    @staticmethod
    def _ready(path: Path) -> bool:
        return path.exists()

    @staticmethod
    def _whisper_v2_runtime_ready() -> bool:
        try:
            importlib.import_module("faster_whisper")
            return True
        except Exception:
            return False
