from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from application.services.prompt_manager import PromptManager
from application.services.web_job_manager import WebJobManager
from presentation.web.schemas import JobStatusView, TranscriptResultView

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

STD_INPUT_HANDLE = -10
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080


def _disable_windows_quick_edit_mode(kernel32: Any | None = None) -> bool:
    """
    Disable QuickEdit on the current console only.

    On classic Windows consoles, a stray mouse selection can freeze the Python
    process until Enter/Esc is pressed. Disabling QuickEdit for this process
    avoids that false "webapp hangs until I press Enter" behavior.
    """
    if os.name != "nt":
        return False

    try:
        kernel32 = kernel32 or ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if handle in (0, -1):
            return False

        current_mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(current_mode)):
            return False

        new_mode = (current_mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        if new_mode == current_mode.value:
            return True

        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception as exc:
        logger.debug("Could not disable QuickEdit mode: %s", exc)
        return False


def _configure_runtime_environment() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    if _disable_windows_quick_edit_mode():
        logger.info("Disabled QuickEdit for the current webapp console.")


def create_app(job_manager: WebJobManager | None = None) -> FastAPI:
    app = FastAPI(title="Hệ thống trinh sát âm thanh", version="1.0.0")
    app.state.job_manager = job_manager or WebJobManager()
    prompt_manager = PromptManager()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "page_title": "Hệ thống trinh sát âm thanh",
                "scenarios": prompt_manager.list_scenarios(),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/models")
    async def get_models() -> dict:
        return app.state.job_manager.get_model_inventory()

    @app.post("/api/jobs", response_model=JobStatusView)
    async def create_job(
        file: UploadFile = File(...),
        asr_engine: str = Form("whisper-v2"),
        analysis_scenario: str = Form("general_intelligence"),
        apply_vad: bool = Form(True),
        apply_hallucination_filter: bool = Form(False),
        apply_domain_postprocess: bool = Form(False),
        domain: str = Form("general"),
        apply_protonx: bool = Form(False),
        apply_llm_correction: bool = Form(False),
        speaker_mode: str = Form("off"),
        speaker_refine: bool = Form(False),
        device: str = Form("cuda"),
    ) -> dict:
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            return app.state.job_manager.create_job(
                filename=file.filename or "upload.wav",
                content=content,
                options_payload={
                    "asr_engine": asr_engine,
                    "analysis_scenario": analysis_scenario,
                    "apply_vad": apply_vad,
                    "apply_hallucination_filter": apply_hallucination_filter,
                    "apply_domain_postprocess": apply_domain_postprocess,
                    "domain": domain,
                    "apply_protonx": apply_protonx,
                    "apply_llm_correction": apply_llm_correction,
                    "speaker_mode": speaker_mode,
                    "speaker_refine": speaker_refine,
                    "device": device,
                },
            )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to create STT job: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}", response_model=JobStatusView)
    async def get_job(job_id: str) -> dict:
        try:
            return app.state.job_manager.get_job_status(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

    @app.get("/api/jobs/{job_id}/result", response_model=TranscriptResultView)
    async def get_result(job_id: str) -> dict:
        try:
            return app.state.job_manager.get_result(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job result not found.") from exc

    @app.post("/api/jobs/{job_id}/steps/{step}", response_model=JobStatusView)
    async def run_step(job_id: str, step: str) -> dict:
        try:
            return app.state.job_manager.enqueue_step(job_id, step)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
    async def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            artifact_path = app.state.job_manager.get_artifact_path(job_id, artifact_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

        if not artifact_path:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(path=artifact_path, filename=artifact_path.name)

    return app


_configure_runtime_environment()
app = create_app()


def main() -> None:
    uvicorn.run("presentation.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
