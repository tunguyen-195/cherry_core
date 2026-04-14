from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from application.services.correction_service import CorrectionService
from application.services.intel_presentation_service import IntelPresentationService
from application.services.prompt_manager import PromptManager
from application.use_cases.generate_report import GenerateStrategicReportUseCase
from core.config import RefinementConfig
from core.domain.entities import SpeakerSegment, Transcript
from core.services.alignment_service import AlignmentService
from core.services.output_formatter import OutputFormatter
from infrastructure.adapters.asr.hallucination_filter import HallucinationFilter
from infrastructure.adapters.correction.vietnamese_postprocessor import VietnamesePostProcessor
from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], None]


@dataclass
class SttJobOptions:
    asr_engine: str = "whisper-v2"
    analysis_scenario: str = "general_intelligence"
    apply_vad: bool = True
    apply_hallucination_filter: bool = False
    apply_domain_postprocess: bool = False
    domain: str = "general"
    apply_protonx: bool = False
    apply_llm_correction: bool = False
    speaker_mode: str = "off"
    speaker_refine: bool = False
    device: str = "cuda"
    requested_device: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SttJobOptions":
        defaults = asdict(cls())
        defaults.update(data or {})
        requested_device = defaults.get("device")

        for key in (
            "apply_vad",
            "apply_hallucination_filter",
            "apply_domain_postprocess",
            "apply_protonx",
            "apply_llm_correction",
            "speaker_refine",
        ):
            defaults[key] = cls._as_bool(defaults.get(key))

        raw_device = defaults.get("device")
        if raw_device is None or str(raw_device).strip() == "":
            normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            normalized_device = str(raw_device).lower()
        if normalized_device not in {"cpu", "cuda"}:
            normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
        if normalized_device == "cuda" and not torch.cuda.is_available():
            normalized_device = "cpu"
        defaults["device"] = normalized_device
        defaults["requested_device"] = str(requested_device).lower() if requested_device is not None else None
        defaults["asr_engine"] = str(defaults.get("asr_engine", "whisper-v2")).lower()
        defaults["analysis_scenario"] = str(defaults.get("analysis_scenario", "general_intelligence")).lower()
        defaults["speaker_mode"] = str(defaults.get("speaker_mode", "off")).lower()
        defaults["domain"] = str(defaults.get("domain", "general")).lower()
        return cls(**defaults)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


class SttWebPipeline:
    """File-oriented orchestration for the offline STT web application."""

    def __init__(self) -> None:
        self.prompt_manager = PromptManager()
        self.intel_presentation_service = IntelPresentationService()

    def run_initial(
        self,
        job_dir: Path,
        state: dict[str, Any],
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        options = SttJobOptions.from_dict(state["options"])
        source_audio = job_dir / state["artifacts"]["source_audio"]

        progress("normalize", 10)
        normalized_audio = job_dir / "normalized.wav"
        self._normalize_audio(source_audio, normalized_audio)
        state["artifacts"]["normalized_audio"] = normalized_audio.name

        working_audio = normalized_audio
        if options.apply_vad:
            progress("vad", 20)
            working_audio = self._maybe_apply_vad(normalized_audio, job_dir, state)

        progress("transcribe", 45)
        transcript = self._transcribe(str(working_audio), options)
        raw_segments = self._normalize_transcript_segments(transcript.segments)
        raw_text = transcript.text.strip()

        state["raw_text"] = raw_text
        state["transcript_segments"] = raw_segments
        state["segments"] = raw_segments
        state.setdefault("metadata", {})
        state["metadata"].update(transcript.metadata)
        state["metadata"]["device"] = options.device
        state["metadata"]["asr_engine"] = options.asr_engine
        state["metadata"]["analysis_scenario"] = options.analysis_scenario

        self._write_text_artifact(job_dir, state, "raw_text", "raw.txt", raw_text)
        self._write_json_artifact(job_dir, state, "transcript_segments", "transcript_segments.json", raw_segments)

        progress("filter", 60)
        filtered_segments = self._filter_transcript_segments(raw_segments, options)
        filtered_text = self._segments_to_text(filtered_segments, raw_text)
        state["filtered_text"] = filtered_text
        state["segments"] = filtered_segments

        self._write_text_artifact(job_dir, state, "filtered_text", "filtered.txt", filtered_text)
        self._write_json_artifact(job_dir, state, "filtered_segments", "filtered_segments.json", filtered_segments)

        if options.apply_protonx:
            state = self.run_step("protonx", job_dir, state, progress)

        if options.apply_llm_correction:
            state = self.run_step("llm_correction", job_dir, state, progress)

        if options.speaker_mode != "off":
            state = self.run_step("diarization", job_dir, state, progress)

        if options.speaker_refine and options.speaker_mode != "off":
            state = self.run_step("speaker_refine", job_dir, state, progress)

        progress("export", 99)
        self._write_result_json(job_dir, state)
        return state

    def run_step(
        self,
        step: str,
        job_dir: Path,
        state: dict[str, Any],
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        options = SttJobOptions.from_dict(state["options"])

        if step == "protonx":
            progress("protonx", 76)
            input_text = self._best_available_text(state)
            from infrastructure.adapters.correction.protonx_adapter import ProtonXAdapter

            corrected = ProtonXAdapter(device=options.device).correct(input_text)
            state["corrected_text"] = corrected or input_text
            state.setdefault("metadata", {}).setdefault("correction_steps", [])
            state["metadata"]["correction_steps"].append("protonx")
            self._write_text_artifact(job_dir, state, "corrected_text", "corrected.txt", state["corrected_text"])

        elif step == "llm_correction":
            progress("llm_correction", 84)
            input_text = self._best_available_text(state)
            corrected = CorrectionService(device=options.device).correct(input_text)
            state["corrected_text"] = corrected or input_text
            state.setdefault("metadata", {}).setdefault("correction_steps", [])
            state["metadata"]["correction_steps"].append("llm_correction")
            self._write_text_artifact(job_dir, state, "corrected_text", "corrected.txt", state["corrected_text"])

        elif step == "intel_summary":
            progress("intel_summary", 88)
            state = self._run_intelligence_summary(job_dir, state, options)

        elif step == "diarization":
            progress("diarization", 92)
            state = self._run_diarization(job_dir, state, options)

        elif step == "stable_ts":
            progress("stable_ts", 68)
            state = self._run_stable_ts(job_dir, state, options)

        elif step == "speaker_refine":
            progress("speaker_refine", 96)
            state = self._run_speaker_refinement(job_dir, state, options)

        else:
            raise ValueError(f"Unsupported step: {step}")

        state.setdefault("completed_steps", [])
        if step not in state["completed_steps"]:
            state["completed_steps"].append(step)
        self._write_result_json(job_dir, state)
        return state

    def _normalize_audio(self, source_audio: Path, output_audio: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
        subprocess.run(command, check=True, capture_output=True)

    def _maybe_apply_vad(self, normalized_audio: Path, job_dir: Path, state: dict[str, Any]) -> Path:
        vad_audio = job_dir / "vad_preprocessed.wav"
        try:
            from infrastructure.adapters.vad.silero_adapter import SileroVADAdapter

            SileroVADAdapter().remove_silence(str(normalized_audio), str(vad_audio))
            state["artifacts"]["vad_audio"] = vad_audio.name
            return vad_audio
        except Exception as exc:
            warning = f"VAD skipped: {exc}"
            state.setdefault("metadata", {}).setdefault("warnings", []).append(warning)
            logger.warning(warning)
            return normalized_audio

    def _transcribe(self, audio_path: str, options: SttJobOptions):
        if options.asr_engine == "phowhisper":
            from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter

            return PhoWhisperAdapter(device=options.device).transcribe(audio_path)
        if options.asr_engine == "whisper-v2":
            from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter

            return WhisperV2Adapter(use_vad=False, device=options.device).transcribe(audio_path)
        if options.asr_engine == "whisperx":
            from infrastructure.adapters.asr.whisperx_adapter import WhisperXAdapter

            compute_type = "float16" if options.device == "cuda" else "int8"
            return WhisperXAdapter(device=options.device, compute_type=compute_type).transcribe(audio_path)
        raise ValueError(f"Unsupported ASR engine: {options.asr_engine}")

    def _run_stable_ts(
        self,
        job_dir: Path,
        state: dict[str, Any],
        options: SttJobOptions,
    ) -> dict[str, Any]:
        stable_audio_name = state.get("artifacts", {}).get("vad_audio") or state.get("artifacts", {}).get("normalized_audio")
        if not stable_audio_name:
            raise RuntimeError("Normalized audio is missing, cannot run Stable-TS.")

        stable_audio = job_dir / stable_audio_name
        from infrastructure.adapters.asr.stablets_adapter import StableTsAdapter

        transcript = StableTsAdapter(device=options.device).transcribe(str(stable_audio))
        stable_segments = self._normalize_transcript_segments(transcript.segments)
        stable_text = transcript.text.strip()

        state["stable_text"] = stable_text
        state["stable_segments"] = stable_segments
        state["segments"] = stable_segments
        state.setdefault("metadata", {}).update(
            {
                "stable_ts_enabled": True,
                "stable_ts_audio": stable_audio.name,
                "stable_ts_backend": transcript.metadata.get("backend", "stable-ts"),
            }
        )
        state["metadata"]["stable_ts_model"] = transcript.metadata.get("model", "stable-ts")

        self._write_text_artifact(job_dir, state, "stable_text", "stable_ts.txt", stable_text)
        self._write_json_artifact(job_dir, state, "stable_segments", "stable_ts_segments.json", stable_segments)
        return state

    def _run_intelligence_summary(
        self,
        job_dir: Path,
        state: dict[str, Any],
        options: SttJobOptions,
    ) -> dict[str, Any]:
        input_text = state.get("speaker_transcript") or self._best_available_text(state)
        if not input_text.strip():
            raise RuntimeError("Không có transcript để tạo tóm tắt trinh sát.")

        llm = LlamaCppAdapter(model_type="vistral", device=options.device)
        report = GenerateStrategicReportUseCase(llm).execute(
            Transcript(text=input_text.strip()),
            options.analysis_scenario,
        )
        report_payload = asdict(report)
        intel_view = self.intel_presentation_service.build(report_payload, input_text, options.analysis_scenario)
        summary_text = self._format_intelligence_report(report_payload, options.analysis_scenario, intel_view)

        state["intel_summary"] = summary_text
        state["intel_report"] = report_payload
        state["intel_cards"] = intel_view.get("intel_cards") or []
        state["intel_timeline"] = intel_view.get("intel_timeline") or []
        state["risk_flags"] = intel_view.get("risk_flags") or []
        state.setdefault("metadata", {})["intel_summary_ready"] = True
        state["metadata"]["intel_summary_source"] = "speaker_transcript" if state.get("speaker_transcript") else "best_available_text"

        self._write_text_artifact(job_dir, state, "intel_summary", "intel_summary.txt", summary_text)
        self._write_json_artifact(job_dir, state, "intel_report", "intel_report.json", report_payload)
        self._write_json_artifact(job_dir, state, "intel_view", "intel_view.json", intel_view)
        return state

    def _run_diarization(
        self,
        job_dir: Path,
        state: dict[str, Any],
        options: SttJobOptions,
    ) -> dict[str, Any]:
        if options.speaker_mode not in {"off", "speechbrain"}:
            raise RuntimeError(f"Unsupported offline speaker mode: {options.speaker_mode}")

        normalized_audio = job_dir / state["artifacts"]["normalized_audio"]
        from infrastructure.adapters.diarization.speechbrain_adapter import SpeechBrainAdapter

        diarizer = SpeechBrainAdapter(use_vad=False)
        speaker_segments = diarizer.diarize(str(normalized_audio))
        speaker_blocks = AlignmentService.align_words(
            state.get("stable_segments") or state.get("segments") or state.get("transcript_segments") or [],
            speaker_segments,
        )

        speaker_blocks = self._filter_speaker_blocks(speaker_blocks, options)
        speaker_transcript = OutputFormatter.format_subtitle_style(speaker_blocks) if speaker_blocks else ""

        state["speaker_segments"] = speaker_blocks
        state["speaker_transcript"] = speaker_transcript
        state.setdefault("metadata", {})["speaker_mode"] = "speechbrain"

        self._write_text_artifact(job_dir, state, "speaker_transcript", "speaker.txt", speaker_transcript)
        self._write_json_artifact(job_dir, state, "speaker_segments", "speaker_segments.json", speaker_blocks)
        return state

    def _run_speaker_refinement(
        self,
        job_dir: Path,
        state: dict[str, Any],
        options: SttJobOptions,
    ) -> dict[str, Any]:
        if not state.get("speaker_segments"):
            raise RuntimeError("Speaker diarization must complete before speaker refinement.")

        from infrastructure.adapters.correction.contextual_refiner import ContextualSpeakerRefiner
        from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter

        llm = LlamaCppAdapter(model_type="vistral", device=options.device)
        if not llm.load():
            raise RuntimeError("Speaker refinement LLM could not be loaded.")

        previous_flag = RefinementConfig.ENABLED
        RefinementConfig.ENABLED = True
        try:
            refiner = ContextualSpeakerRefiner(llm_adapter=llm)
            refined = refiner.refine([self._block_to_speaker_segment(block) for block in state["speaker_segments"]])
        finally:
            RefinementConfig.ENABLED = previous_flag

        speaker_blocks = [self._speaker_segment_to_block(segment) for segment in refined]
        state["speaker_segments"] = speaker_blocks
        state["speaker_transcript"] = OutputFormatter.format_subtitle_style(speaker_blocks)
        state.setdefault("metadata", {})["speaker_refined"] = True

        self._write_text_artifact(job_dir, state, "speaker_transcript", "speaker.txt", state["speaker_transcript"])
        self._write_json_artifact(job_dir, state, "speaker_segments", "speaker_segments.json", speaker_blocks)
        return state

    def _filter_transcript_segments(
        self,
        transcript_segments: list[dict[str, Any]],
        options: SttJobOptions,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        processor = self._build_postprocessor(options)
        for segment in transcript_segments:
            text = str(segment.get("text", "")).strip()
            if options.apply_hallucination_filter:
                text = HallucinationFilter.filter(text, language="vi")
            if processor:
                text = processor.process(text)
            if not text:
                continue

            cleaned = dict(segment)
            cleaned["text"] = text
            filtered.append(cleaned)
        return filtered

    def _filter_speaker_blocks(
        self,
        speaker_blocks: list[dict[str, Any]],
        options: SttJobOptions,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        processor = self._build_postprocessor(options)
        for block in speaker_blocks:
            text = str(block.get("text", "")).strip()
            if options.apply_hallucination_filter:
                text = HallucinationFilter.filter(text, language="vi")
            if processor:
                text = processor.process(text)
            if not text:
                continue

            cleaned = dict(block)
            cleaned["text"] = text
            filtered.append(cleaned)
        return filtered

    def _build_postprocessor(self, options: SttJobOptions) -> VietnamesePostProcessor | None:
        if not options.apply_domain_postprocess:
            return None
        return VietnamesePostProcessor(domain=options.domain)

    def _normalize_transcript_segments(self, segments: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for segment in segments or []:
            if isinstance(segment, dict):
                normalized.append(
                    {
                        "start": float(segment.get("start", 0)),
                        "end": float(segment.get("end", 0)),
                        "text": str(segment.get("text", "")),
                        "words": segment.get("words", []),
                    }
                )
                continue

            normalized.append(
                {
                    "start": float(getattr(segment, "start", 0)),
                    "end": float(getattr(segment, "end", 0)),
                    "text": str(getattr(segment, "text", "")),
                    "words": getattr(segment, "words", []),
                }
            )
        return normalized

    def _segments_to_text(self, segments: list[dict[str, Any]], fallback: str) -> str:
        text = " ".join(segment.get("text", "").strip() for segment in segments if segment.get("text", "").strip()).strip()
        return text or fallback

    def _best_available_text(self, state: dict[str, Any]) -> str:
        return (
            state.get("speaker_transcript")
            or state.get("corrected_text")
            or state.get("stable_text")
            or state.get("filtered_text")
            or state.get("raw_text")
            or ""
        )

    def _format_intelligence_report(
        self,
        report: dict[str, Any],
        scenario: str,
        intel_view: dict[str, Any] | None = None,
    ) -> str:
        strategic = report.get("strategic_assessment") or {}
        conclusion = strategic.get("final_conclusion") or {}
        recommendations = report.get("operational_recommendations") or []
        intel_view = intel_view or {}
        cards = {card.get("id"): card for card in intel_view.get("intel_cards") or [] if isinstance(card, dict)}
        scenario_info = self.prompt_manager.load_scenario(scenario)
        scenario_label = scenario_info.get("name") or scenario_info.get("scenario_name") or scenario

        return self.prompt_manager.render_template(
            "forensic_brief.j2",
            scenario_label=scenario_label,
            threat_level=strategic.get("threat_level", "UNKNOWN"),
            classification=strategic.get("classification", "Không có thông tin"),
            executive_briefing=strategic.get("executive_briefing", "Không có thông tin"),
            verdict=conclusion.get("verdict", "Không có kết luận"),
            investigator_note=conclusion.get("investigator_note", ""),
            risk_flags=intel_view.get("risk_flags") or [],
            subject_items=(cards.get("subjects") or {}).get("items") or [],
            location_items=(cards.get("locations") or {}).get("items") or [],
            sensitive_items=(cards.get("sensitive") or {}).get("items") or [],
            finance_items=(cards.get("financial") or {}).get("items") or [],
            slang_items=(cards.get("slang") or {}).get("items") or [],
            timeline_items=intel_view.get("intel_timeline") or [],
            recommendations=recommendations,
        ).strip()

    def _speaker_segment_to_block(self, segment: SpeakerSegment) -> dict[str, Any]:
        return {
            "speaker": segment.speaker_id,
            "text": segment.text.strip(),
            "start": segment.start_time,
            "end": segment.end_time,
            "words": segment.words,
        }

    def _block_to_speaker_segment(self, block: dict[str, Any]) -> SpeakerSegment:
        return SpeakerSegment(
            start_time=float(block.get("start", 0)),
            end_time=float(block.get("end", 0)),
            speaker_id=str(block.get("speaker", "UNKNOWN")),
            text=str(block.get("text", "")),
            words=block.get("words", []),
        )

    def _write_text_artifact(
        self,
        job_dir: Path,
        state: dict[str, Any],
        artifact_key: str,
        filename: str,
        content: str,
    ) -> None:
        artifact_path = job_dir / filename
        artifact_path.write_text(content or "", encoding="utf-8")
        state.setdefault("artifacts", {})[artifact_key] = filename

    def _write_json_artifact(
        self,
        job_dir: Path,
        state: dict[str, Any],
        artifact_key: str,
        filename: str,
        payload: Any,
    ) -> None:
        import json

        artifact_path = job_dir / filename
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        state.setdefault("artifacts", {})[artifact_key] = filename

    def _write_result_json(self, job_dir: Path, state: dict[str, Any]) -> None:
        payload = {
            "job_id": state["job_id"],
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
            "artifacts": state.get("artifacts") or {},
        }
        self._write_json_artifact(job_dir, state, "result_json", "result.json", payload)
