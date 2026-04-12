from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from presentation.web.app import create_app


class FakeJobManager:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.status = {
            "job_id": "job-123",
            "state": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": "2026-04-06T00:00:00Z",
            "updated_at": "2026-04-06T00:00:00Z",
            "error": None,
            "artifacts": ["raw.txt"],
        }
        self.result = {
            "job_id": "job-123",
            "language": "vi",
            "raw_text": "xin chao",
            "stable_text": "xin chào ổn định",
            "filtered_text": "xin chào",
            "corrected_text": "Xin chào",
            "intel_summary": "Mức độ đe dọa: LOW",
            "intel_report": {
                "strategic_assessment": {
                    "executive_briefing": "Cuộc hội thoại mang tính thu thập thông tin.",
                    "threat_level": "LOW",
                    "classification": "Trinh sát tổng hợp",
                    "final_conclusion": {
                        "verdict": "Cần theo dõi",
                        "investigator_note": "Chưa thấy dấu hiệu nguy hiểm tức thời.",
                    },
                },
                "operational_recommendations": ["Theo dõi thêm lịch trình trong 48 giờ tới."],
            },
            "intel_cards": [
                {
                    "id": "subjects",
                    "title": "Chủ thể và liên hệ",
                    "summary": "Các đối tượng nổi bật",
                    "items": [{"label": "Đối tượng A", "value": "Người bị theo dõi", "meta": ""}],
                }
            ],
            "intel_timeline": [
                {
                    "time": "08:00",
                    "title": "Liên lạc lần đầu",
                    "location": "Hà Nội",
                    "detail": "Hẹn gặp",
                    "actors": ["Đối tượng A"],
                }
            ],
            "risk_flags": [{"level": "low", "label": "Theo dõi nền", "detail": "Chưa có dấu hiệu escalated."}],
            "speaker_transcript": "[Speaker 1] Xin chào",
            "segments": [{"start": 0, "end": 1, "text": "xin chào"}],
            "stable_segments": [{"start": 0, "end": 1, "text": "xin chào ổn định"}],
            "speaker_segments": [{"speaker": "SPEAKER_1", "start": 0, "end": 1, "text": "xin chào"}],
            "metadata": {"device": "cpu", "asr_engine": "whisper-v2", "analysis_scenario": "general_intelligence"},
            "downloads": {"raw_text": "/api/jobs/job-123/artifacts/raw.txt"},
        }
        self.inventory = {
            "items": [
                {"model_id": "phowhisper", "family": "asr", "offline_ready": True, "notes": "", "path": "", "load_status": "unloaded"}
            ],
            "capabilities": {
                "asr_engines": {"phowhisper": True, "whisper-v2": True, "whisperx": False},
                "speaker_modes": {"off": True, "speechbrain": True},
                "features": {"apply_vad": True, "apply_stable_ts": True, "apply_protonx": True, "apply_llm_correction": True, "apply_intel_summary": True, "speaker_refine": True},
                "devices": {"cpu": True, "cuda": False},
            },
        }
        self.artifact_path = tmp_path / "raw.txt"
        self.artifact_path.write_text("artifact", encoding="utf-8")

    def create_job(self, filename: str, content: bytes, options_payload: dict):
        assert filename == "sample.wav"
        assert content == b"abc"
        assert options_payload["analysis_scenario"] == "general_intelligence"
        self.status["state"] = "queued"
        self.status["stage"] = "queued"
        return dict(self.status)

    def enqueue_step(self, job_id: str, step: str):
        assert job_id == "job-123"
        self.status["state"] = "queued"
        self.status["stage"] = step
        return dict(self.status)

    def get_job_status(self, job_id: str):
        if job_id != "job-123":
            raise FileNotFoundError(job_id)
        return dict(self.status)

    def get_result(self, job_id: str):
        if job_id != "job-123":
            raise FileNotFoundError(job_id)
        return dict(self.result)

    def get_model_inventory(self):
        return self.inventory

    def get_artifact_path(self, job_id: str, artifact_name: str):
        if job_id != "job-123":
            raise FileNotFoundError(job_id)
        if artifact_name != "raw.txt":
            return None
        return self.artifact_path


def test_webapp_routes(tmp_path: Path):
    client = TestClient(create_app(job_manager=FakeJobManager(tmp_path)))

    index = client.get("/")
    assert index.status_code == 200
    assert "Hệ thống trinh sát âm thanh" in index.text
    assert "Giảm ảo giác ngữ cảnh" in index.text

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 204

    inventory = client.get("/api/models")
    assert inventory.status_code == 200
    assert inventory.json()["capabilities"]["asr_engines"]["phowhisper"] is True

    created = client.post(
        "/api/jobs",
        files={"file": ("sample.wav", b"abc", "audio/wav")},
        data={
            "asr_engine": "phowhisper",
            "analysis_scenario": "general_intelligence",
            "apply_vad": "true",
            "apply_hallucination_filter": "true",
            "apply_domain_postprocess": "false",
            "domain": "general",
            "apply_protonx": "false",
            "apply_llm_correction": "false",
            "speaker_mode": "off",
            "speaker_refine": "false",
            "device": "cpu",
        },
    )
    assert created.status_code == 200
    assert created.json()["job_id"] == "job-123"

    status = client.get("/api/jobs/job-123")
    assert status.status_code == 200
    assert status.json()["stage"] == "queued"

    result = client.get("/api/jobs/job-123/result")
    assert result.status_code == 200
    assert result.json()["corrected_text"] == "Xin chào"
    assert result.json()["intel_report"]["strategic_assessment"]["threat_level"] == "LOW"
    assert result.json()["intel_cards"][0]["id"] == "subjects"
    assert result.json()["intel_timeline"][0]["title"] == "Liên lạc lần đầu"
    assert result.json()["risk_flags"][0]["level"] == "low"

    step = client.post("/api/jobs/job-123/steps/protonx")
    assert step.status_code == 200
    assert step.json()["stage"] == "protonx"

    stable_step = client.post("/api/jobs/job-123/steps/stable_ts")
    assert stable_step.status_code == 200
    assert stable_step.json()["stage"] == "stable_ts"

    intel_step = client.post("/api/jobs/job-123/steps/intel_summary")
    assert intel_step.status_code == 200
    assert intel_step.json()["stage"] == "intel_summary"

    artifact = client.get("/api/jobs/job-123/artifacts/raw.txt")
    assert artifact.status_code == 200
    assert artifact.text == "artifact"
