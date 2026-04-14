from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from application.services.model_inventory_service import ModelInventoryService
from application.services.stt_web_pipeline import SttJobOptions, SttWebPipeline
from core.config import BASE_DIR

logger = logging.getLogger(__name__)


class WebJobManager:
    """Single-worker file-based job manager for the offline STT webapp."""

    def __init__(
        self,
        jobs_root: Path | None = None,
        pipeline: SttWebPipeline | None = None,
        inventory_service: ModelInventoryService | None = None,
    ) -> None:
        self.jobs_root = jobs_root or (BASE_DIR / "output" / "webapp" / "jobs")
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline or SttWebPipeline()
        self.inventory_service = inventory_service or ModelInventoryService()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cherry-web")
        self.lock = threading.Lock()
        self._recover_interrupted_jobs()

    def create_job(self, filename: str, content: bytes, options_payload: dict[str, Any]) -> dict[str, Any]:
        options = SttJobOptions.from_dict(options_payload)
        self._validate_options(options)

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        suffix = Path(filename).suffix or ".wav"
        source_name = f"source{suffix}"
        (job_dir / source_name).write_bytes(content)

        now = self._now()
        job_status = {
            "job_id": job_id,
            "state": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifacts": [source_name],
        }
        state = {
            "job_id": job_id,
            "source_filename": filename,
            "options": options.to_dict(),
            "created_at": now,
            "metadata": {"warnings": []},
            "artifacts": {"source_audio": source_name},
            "completed_steps": [],
            "raw_text": None,
            "stable_text": None,
            "filtered_text": None,
            "corrected_text": None,
            "intel_summary": None,
            "intel_report": {},
            "intel_cards": [],
            "intel_timeline": [],
            "risk_flags": [],
            "speaker_transcript": None,
            "transcript_segments": [],
            "stable_segments": [],
            "segments": [],
            "speaker_segments": [],
        }
        requested_device = options_payload.get("device")
        if str(requested_device).lower() == "cuda" and options.device == "cpu":
            state["metadata"]["warnings"].append("CUDA không khả dụng; hệ thống tự chuyển sang CPU.")

        self._write_json(self._job_file(job_id), job_status)
        self._write_json(self._state_file(job_id), state)
        self.executor.submit(self._run_initial_job, job_id)
        return job_status

    def enqueue_step(self, job_id: str, step: str) -> dict[str, Any]:
        with self.lock:
            job = self._read_json(self._job_file(job_id))
            state = self._read_json(self._state_file(job_id))

            if job["state"] in {"queued", "running"}:
                raise RuntimeError("Job is already in progress.")

            options = SttJobOptions.from_dict(state["options"])
            self._validate_step(step, state, options)
            self._set_status(job_id, "queued", step, job.get("progress", 0), None)
            self.executor.submit(self._run_step_job, job_id, step)
            return self.get_job_status(job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._read_json(self._job_file(job_id))

    def get_result(self, job_id: str) -> dict[str, Any]:
        state = self._read_json(self._state_file(job_id))
        artifacts = state.get("artifacts", {})
        downloads = {
            key: f"/api/jobs/{job_id}/artifacts/{filename}"
            for key, filename in artifacts.items()
            if filename and (self.jobs_root / job_id / filename).exists()
        }
        return {
            "job_id": job_id,
            "language": "vi",
            "raw_text": state.get("raw_text"),
            "stable_text": state.get("stable_text"),
            "filtered_text": state.get("filtered_text"),
            "corrected_text": state.get("corrected_text"),
            "intel_summary": state.get("intel_summary"),
            "intel_report": state.get("intel_report") or {},
            "intel_cards": state.get("intel_cards") or [],
            "intel_timeline": state.get("intel_timeline") or [],
            "risk_flags": state.get("risk_flags") or [],
            "speaker_transcript": state.get("speaker_transcript"),
            "segments": state.get("segments") or [],
            "stable_segments": state.get("stable_segments") or [],
            "speaker_segments": state.get("speaker_segments") or [],
            "metadata": state.get("metadata") or {},
            "downloads": downloads,
        }

    def get_model_inventory(self) -> dict[str, Any]:
        return self.inventory_service.get_inventory()

    def get_artifact_path(self, job_id: str, artifact_name: str) -> Path | None:
        state = self._read_json(self._state_file(job_id))
        allowed_names = set((state.get("artifacts") or {}).values())
        if artifact_name not in allowed_names:
            return None
        artifact_path = self.jobs_root / job_id / artifact_name
        return artifact_path if artifact_path.exists() else None

    def _run_initial_job(self, job_id: str) -> None:
        try:
            self._set_status(job_id, "running", "starting", 1, None)
            state = self._read_json(self._state_file(job_id))
            job_dir = self.jobs_root / job_id

            def progress(stage: str, value: int) -> None:
                self._set_status(job_id, "running", stage, value, None)

            state = self.pipeline.run_initial(job_dir, state, progress)
            state.setdefault("completed_steps", [])
            if "base" not in state["completed_steps"]:
                state["completed_steps"].insert(0, "base")
            self._write_json(self._state_file(job_id), state)
            self._set_status(job_id, "completed", "completed", 100, None, artifacts=self._artifact_list(state))
        except Exception as exc:
            logger.exception("Initial web job failed: %s", exc)
            self._set_status(job_id, "failed", "failed", 100, str(exc))

    def _run_step_job(self, job_id: str, step: str) -> None:
        try:
            self._set_status(job_id, "running", step, 1, None)
            state = self._read_json(self._state_file(job_id))
            job_dir = self.jobs_root / job_id

            def progress(stage: str, value: int) -> None:
                self._set_status(job_id, "running", stage, value, None)

            state = self.pipeline.run_step(step, job_dir, state, progress)
            self._write_json(self._state_file(job_id), state)
            self._set_status(job_id, "completed", "completed", 100, None, artifacts=self._artifact_list(state))
        except Exception as exc:
            logger.exception("Web job step failed: %s", exc)
            self._set_status(job_id, "failed", "failed", 100, str(exc))

    def _validate_options(self, options: SttJobOptions) -> None:
        inventory = self.get_model_inventory()["capabilities"]
        if not inventory["asr_engines"].get(options.asr_engine):
            raise RuntimeError(f"ASR engine '{options.asr_engine}' is not available offline.")
        if options.speaker_mode != "off" and not inventory["speaker_modes"].get(options.speaker_mode):
            raise RuntimeError(f"Speaker mode '{options.speaker_mode}' is not available.")
        if options.apply_protonx and not inventory["features"].get("apply_protonx"):
            raise RuntimeError("ProtonX correction model is not available.")
        if options.apply_llm_correction and not inventory["features"].get("apply_llm_correction"):
            raise RuntimeError("Local LLM correction model is not available.")
        if options.speaker_refine and not inventory["features"].get("speaker_refine"):
            raise RuntimeError("Speaker refinement model is not available.")
        if options.apply_vad and not inventory["features"].get("apply_vad"):
            raise RuntimeError("Silero VAD is not available.")

    def _validate_step(self, step: str, state: dict[str, Any], options: SttJobOptions) -> None:
        self._validate_options(options)
        if step not in {"stable_ts", "protonx", "llm_correction", "intel_summary", "diarization", "speaker_refine"}:
            raise RuntimeError(f"Unsupported step: {step}")
        if not state.get("raw_text"):
            raise RuntimeError("Base transcription must complete before running extra steps.")
        if step == "stable_ts":
            if not self.get_model_inventory()["capabilities"]["features"].get("apply_stable_ts"):
                raise RuntimeError("Stable-TS offline package/model is not available.")
            if options.asr_engine != "whisper-v2":
                raise RuntimeError("Stable-TS step currently supports jobs created with ASR engine 'whisper-v2' only.")
        if step == "intel_summary" and not self.get_model_inventory()["capabilities"]["features"].get("apply_intel_summary"):
            raise RuntimeError("Hệ phân tích trinh sát cục bộ chưa sẵn sàng.")
        if step == "speaker_refine" and not state.get("speaker_segments"):
            raise RuntimeError("Speaker refinement requires diarization output.")

    def _recover_interrupted_jobs(self) -> None:
        for job_dir in self.jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            job_file = job_dir / "job.json"
            if not job_file.exists():
                continue
            job = self._read_json(job_file)
            if job.get("state") in {"queued", "running"}:
                job["state"] = "failed"
                job["stage"] = "failed"
                job["error"] = "Previous server instance stopped before job completed."
                job["updated_at"] = self._now()
                self._write_json(job_file, job)

    def _set_status(
        self,
        job_id: str,
        state: str,
        stage: str,
        progress: int,
        error: str | None,
        artifacts: list[str] | None = None,
    ) -> None:
        with self.lock:
            job = self._read_json(self._job_file(job_id))
            job["state"] = state
            job["stage"] = stage
            job["progress"] = progress
            job["error"] = error
            job["updated_at"] = self._now()
            if artifacts is not None:
                job["artifacts"] = artifacts
            self._write_json(self._job_file(job_id), job)

    def _artifact_list(self, state: dict[str, Any]) -> list[str]:
        return sorted({name for name in (state.get("artifacts") or {}).values() if name})

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "job.json"

    def _state_file(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "state.json"

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"
