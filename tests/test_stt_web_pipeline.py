from __future__ import annotations

from pathlib import Path

from application.services.intel_presentation_service import IntelPresentationService
from application.services.stt_web_pipeline import SttJobOptions, SttWebPipeline
from core.domain.entities import StrategicReport, Transcript


def test_run_stable_ts_step_updates_state_with_stable_output(tmp_path: Path, monkeypatch):
    job_dir = tmp_path
    normalized_audio = job_dir / "normalized.wav"
    normalized_audio.write_bytes(b"wav")

    class FakeStableTsAdapter:
        def __init__(self, device: str):
            assert device == "cpu"

        def transcribe(self, audio_path: str) -> Transcript:
            assert audio_path == str(normalized_audio)
            return Transcript(
                text="xin chào ổn định",
                segments=[{"start": 0.0, "end": 1.0, "text": "xin chào ổn định", "words": []}],
                metadata={"backend": "stable-ts+faster-whisper", "model": "stable-ts"},
            )

    monkeypatch.setattr(
        "infrastructure.adapters.asr.stablets_adapter.StableTsAdapter",
        FakeStableTsAdapter,
    )

    state = {
        "job_id": "job-1",
        "options": {"asr_engine": "whisper-v2", "device": "cpu"},
        "artifacts": {"normalized_audio": "normalized.wav"},
        "metadata": {},
        "completed_steps": [],
        "raw_text": "xin chao",
        "stable_text": None,
        "stable_segments": [],
        "segments": [{"start": 0.0, "end": 1.0, "text": "xin chao", "words": []}],
    }

    result = SttWebPipeline().run_step("stable_ts", job_dir, state, lambda _stage, _value: None)

    assert result["stable_text"] == "xin chào ổn định"
    assert result["segments"][0]["text"] == "xin chào ổn định"
    assert result["stable_segments"][0]["text"] == "xin chào ổn định"
    assert result["metadata"]["stable_ts_enabled"] is True
    assert result["metadata"]["stable_ts_backend"] == "stable-ts+faster-whisper"
    assert "stable_ts" in result["completed_steps"]


def test_run_intel_summary_step_creates_human_readable_brief(tmp_path: Path, monkeypatch):
    job_dir = tmp_path

    class FakeLlamaCppAdapter:
        def __init__(self, model_type: str, device: str):
            assert model_type == "vistral"
            assert device == "cpu"

    class FakeUseCase:
        def __init__(self, llm_engine):
            assert isinstance(llm_engine, FakeLlamaCppAdapter)

        def execute(self, transcript: Transcript, scenario: str) -> StrategicReport:
            assert transcript.text == "[SPEAKER_1] theo dõi đối tượng tại bến xe"
            assert scenario == "general_intelligence"
            return StrategicReport(
                strategic_assessment={
                    "executive_briefing": "Đối tượng đề cập hoạt động theo dõi tại bến xe.",
                    "threat_level": "MEDIUM",
                    "classification": "Trinh sát tổng hợp",
                    "final_conclusion": {
                        "verdict": "Cần theo dõi",
                        "investigator_note": "Nên giữ giám sát mềm, chưa cần can thiệp.",
                    },
                },
                tactical_intelligence={
                    "intelligence_5w1h": {
                        "people": [{"name": "Đối tượng A", "role": "Người bị theo dõi"}],
                        "events": [{"action": "theo dõi", "time": "sáng nay", "location": "bến xe"}],
                    },
                },
                behavioral_profiling={},
                operational_recommendations=["Tiếp tục đối chiếu lịch trình 24 giờ tới."],
            )

    monkeypatch.setattr(
        "application.services.stt_web_pipeline.LlamaCppAdapter",
        FakeLlamaCppAdapter,
    )
    monkeypatch.setattr(
        "application.services.stt_web_pipeline.GenerateStrategicReportUseCase",
        FakeUseCase,
    )

    state = {
        "job_id": "job-2",
        "options": {"asr_engine": "whisper-v2", "device": "cpu", "analysis_scenario": "general_intelligence"},
        "artifacts": {},
        "metadata": {"analysis_scenario": "general_intelligence"},
        "completed_steps": [],
        "raw_text": "theo doi doi tuong",
        "speaker_transcript": "[SPEAKER_1] theo dõi đối tượng tại bến xe",
        "intel_summary": None,
        "intel_report": {},
    }

    result = SttWebPipeline().run_step("intel_summary", job_dir, state, lambda _stage, _value: None)

    assert "Mức độ đe dọa: MEDIUM" in result["intel_summary"]
    assert "Đối tượng A" in result["intel_summary"]
    assert result["intel_report"]["strategic_assessment"]["classification"] == "Trinh sát tổng hợp"
    assert result["intel_cards"][0]["id"] == "subjects"
    assert result["intel_timeline"][0]["title"] == "theo dõi"
    assert result["risk_flags"][0]["level"] in {"medium", "high", "critical", "low"}
    assert result["metadata"]["intel_summary_ready"] is True
    assert result["metadata"]["intel_summary_source"] == "speaker_transcript"
    assert "intel_summary" in result["completed_steps"]


def test_intel_presentation_service_extracts_cards_timeline_and_risks():
    service = IntelPresentationService()
    report = {
        "strategic_assessment": {
            "threat_level": "HIGH",
            "classification": "Theo dõi đối tượng",
            "final_conclusion": {"verdict": "Nghi vấn cao", "investigator_note": "Cần giám sát mềm."},
        },
        "tactical_intelligence": {
            "intelligence_5w1h": {
                "people": [{"name": "Nguyễn Văn A", "role": "Người bị theo dõi"}],
                "events": [{"action": "gặp mặt", "time": "19 giờ", "location": "bến xe Mỹ Đình", "actors": ["Nguyễn Văn A"]}],
            },
            "quantitative_data": {"financials": [{"amount": "5 triệu", "currency": "VND", "context": "tiền giao hàng"}]},
            "sensitive_info": {"pii_detected": [{"type": "phone", "value": "0912345678", "owner": "Nguyễn Văn A"}]},
        },
        "behavioral_profiling": {"psychological_profile_vn": {"risk_level": "high"}},
        "operational_recommendations": ["Theo dõi thêm 24 giờ."],
    }
    transcript = "Anh A hẹn 19 giờ ở bến xe Mỹ Đình, gọi số 0912345678 để nhận 5 triệu tiền giao hàng."

    result = service.build(report, transcript, "general_intelligence")

    assert result["intel_cards"]
    assert any(card["id"] == "subjects" for card in result["intel_cards"])
    assert any(item["value"] == "Người bị theo dõi" for card in result["intel_cards"] for item in card["items"])
    assert result["intel_timeline"][0]["time"] == "19 giờ"
    assert any(flag["level"] == "high" for flag in result["risk_flags"])


def test_stt_job_options_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr("application.services.stt_web_pipeline.torch.cuda.is_available", lambda: False)

    options = SttJobOptions.from_dict({"device": "cuda"})

    assert options.device == "cpu"
    assert options.requested_device == "cuda"


def test_stt_job_options_defaults_to_whisper_v2():
    options = SttJobOptions.from_dict({})

    assert options.asr_engine == "whisper-v2"
