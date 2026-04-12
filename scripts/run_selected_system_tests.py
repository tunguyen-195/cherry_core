from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from application.services.stt_web_pipeline import SttWebPipeline
from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter

sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = ROOT_DIR / "data" / "datasets" / "benchmark_vi_longform_v1"
OUTPUT_DIR = ROOT_DIR / "output" / "benchmarks"
Token = TypeVar("Token")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    audio_name: str
    transcript_name: str
    function_under_test: str
    rationale: str

    @property
    def audio_path(self) -> Path:
        return DATASET_DIR / self.audio_name

    @property
    def transcript_path(self) -> Path:
        return DATASET_DIR / self.transcript_name


@dataclass
class CaseResult:
    case_id: str
    title: str
    function_under_test: str
    audio_name: str
    transcript_name: str
    status: str
    elapsed_sec: float
    metrics: dict[str, Any]
    notes: list[str]


SELECTED_CASES = [
    CaseSpec(
        case_id="short_clean_asr",
        title="Short Clean Vietnamese ASR Accuracy",
        audio_name="audio_01.wav",
        transcript_name="audio_01.txt",
        function_under_test="phowhisper_short_form_asr",
        rationale="Short 4.94s clip, clean single-speaker sentence with full Vietnamese diacritics.",
    ),
    CaseSpec(
        case_id="vad_preprocessing",
        title="Silero VAD Preprocessing Availability",
        audio_name="audio_09.wav",
        transcript_name="audio_09.txt",
        function_under_test="silero_vad_preprocessing",
        rationale="Short 5.38s clip used to isolate the VAD step without long ASR runtime.",
    ),
    CaseSpec(
        case_id="longform_chunking",
        title="Long-form Chunk Stitching Stability",
        audio_name="audio_20.wav",
        transcript_name="audio_20.txt",
        function_under_test="phowhisper_long_form_chunking",
        rationale="Canonical 182.82s benchmark-slice sample that exposes overlap and duplication behavior.",
    ),
]


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()
    return normalized


def levenshtein_distance(source: list[Token], target: list[Token]) -> int:
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


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    reference_words = normalize_text(reference_text).split()
    hypothesis_words = normalize_text(hypothesis_text).split()
    return levenshtein_distance(reference_words, hypothesis_words) / max(1, len(reference_words))


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    reference_chars = list(normalize_text(reference_text).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis_text).replace(" ", ""))
    return levenshtein_distance(reference_chars, hypothesis_chars) / max(1, len(reference_chars))


def count_local_repeated_ngrams(words: list[str], ngram_size: int = 8, max_gap: int = 30) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if len(words) < ngram_size * 2:
        return hits

    last_start = len(words) - ngram_size + 1
    for start in range(last_start):
        gram = tuple(words[start : start + ngram_size])
        search_end = min(last_start, start + max_gap + 1)
        for repeat_start in range(start + ngram_size, search_end):
            if tuple(words[repeat_start : repeat_start + ngram_size]) == gram:
                hits.append(
                    {
                        "first_index": start,
                        "repeat_index": repeat_start,
                        "phrase": " ".join(gram),
                    }
                )
                break
    return hits


def read_wave_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def run_short_clean_asr_case(case: CaseSpec, adapter: PhoWhisperAdapter) -> CaseResult:
    reference_text = case.transcript_path.read_text(encoding="utf-8")
    start = time.time()
    transcript = adapter.transcribe(str(case.audio_path))
    elapsed = time.time() - start

    word_error_rate = compute_wer(reference_text, transcript.text)
    char_error_rate = compute_cer(reference_text, transcript.text)
    reference_words = len(normalize_text(reference_text).split())
    hypothesis_words = len(normalize_text(transcript.text).split())

    status = "pass" if word_error_rate <= 0.05 and char_error_rate <= 0.03 else "fail"
    notes = [
        "Expected behavior: short clean clip should transcribe almost exactly.",
    ]
    if status == "pass":
        notes.append("Observed output is effectively exact after normalization.")
    else:
        notes.append("Short-form ASR drift is higher than expected for a clean sentence.")

    return CaseResult(
        case_id=case.case_id,
        title=case.title,
        function_under_test=case.function_under_test,
        audio_name=case.audio_name,
        transcript_name=case.transcript_name,
        status=status,
        elapsed_sec=round(elapsed, 2),
        metrics={
            "audio_duration_sec": round(read_wave_duration(case.audio_path), 2),
            "wer": round(word_error_rate, 4),
            "cer": round(char_error_rate, 4),
            "reference_word_count": reference_words,
            "hypothesis_word_count": hypothesis_words,
            "predicted_text": transcript.text,
        },
        notes=notes,
    )


def run_vad_preprocessing_case(case: CaseSpec) -> CaseResult:
    pipeline = SttWebPipeline()
    state = {"metadata": {"warnings": []}, "artifacts": {}}

    with tempfile.TemporaryDirectory(prefix="cherry_vad_", dir=str(OUTPUT_DIR)) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        normalized_audio = temp_dir / "normalized.wav"

        pipeline._normalize_audio(case.audio_path, normalized_audio)
        start = time.time()
        working_audio = pipeline._maybe_apply_vad(normalized_audio, temp_dir, state)
        elapsed = time.time() - start

        warnings = list(state.get("metadata", {}).get("warnings", []))
        processed_duration = read_wave_duration(working_audio)
        normalized_duration = read_wave_duration(normalized_audio)

    vad_executed = working_audio.name != normalized_audio.name and not warnings
    fallback_safe = working_audio.name == normalized_audio.name and bool(warnings)
    status = "pass" if vad_executed else "fail"

    notes = [
        "Expected behavior: VAD should run offline and produce a preprocessed waveform.",
    ]
    if fallback_safe:
        notes.append("System fallback is safe: pipeline preserved the normalized audio instead of crashing.")
    if warnings:
        notes.extend(warnings)

    return CaseResult(
        case_id=case.case_id,
        title=case.title,
        function_under_test=case.function_under_test,
        audio_name=case.audio_name,
        transcript_name=case.transcript_name,
        status=status,
        elapsed_sec=round(elapsed, 2),
        metrics={
            "audio_duration_sec": round(read_wave_duration(case.audio_path), 2),
            "normalized_duration_sec": round(normalized_duration, 2),
            "processed_duration_sec": round(processed_duration, 2),
            "vad_executed": vad_executed,
            "fallback_safe": fallback_safe,
            "warning_count": len(warnings),
        },
        notes=notes,
    )


def run_longform_chunking_case(case: CaseSpec, adapter: PhoWhisperAdapter) -> CaseResult:
    reference_text = case.transcript_path.read_text(encoding="utf-8")
    start = time.time()
    transcript = adapter.transcribe(str(case.audio_path))
    elapsed = time.time() - start

    normalized_reference_words = normalize_text(reference_text).split()
    normalized_hypothesis_words = normalize_text(transcript.text).split()
    word_error_rate = compute_wer(reference_text, transcript.text)
    char_error_rate = compute_cer(reference_text, transcript.text)
    word_count_ratio = len(normalized_hypothesis_words) / max(1, len(normalized_reference_words))
    repeated_ngrams = count_local_repeated_ngrams(normalized_hypothesis_words, ngram_size=8, max_gap=30)

    status = (
        "pass"
        if word_error_rate <= 0.18 and char_error_rate <= 0.15 and word_count_ratio <= 1.03 and not repeated_ngrams
        else "fail"
    )
    notes = [
        "Expected behavior: long-form output should not duplicate overlap regions.",
    ]
    if repeated_ngrams:
        notes.append(f"Detected {len(repeated_ngrams)} locally repeated 8-grams, which indicates chunk overlap duplication.")
    if word_count_ratio > 1.03:
        notes.append("Hypothesis is longer than reference beyond the allowed margin.")

    return CaseResult(
        case_id=case.case_id,
        title=case.title,
        function_under_test=case.function_under_test,
        audio_name=case.audio_name,
        transcript_name=case.transcript_name,
        status=status,
        elapsed_sec=round(elapsed, 2),
        metrics={
            "audio_duration_sec": round(read_wave_duration(case.audio_path), 2),
            "wer": round(word_error_rate, 4),
            "cer": round(char_error_rate, 4),
            "reference_word_count": len(normalized_reference_words),
            "hypothesis_word_count": len(normalized_hypothesis_words),
            "word_count_ratio": round(word_count_ratio, 4),
            "repeated_8gram_hits": len(repeated_ngrams),
            "first_repeated_phrase": repeated_ngrams[0]["phrase"] if repeated_ngrams else None,
            "predicted_preview": transcript.text[:700],
        },
        notes=notes,
    )


def build_report(results: list[CaseResult], device: str) -> dict[str, Any]:
    passed = sum(1 for result in results if result.status == "pass")
    failed = sum(1 for result in results if result.status == "fail")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(DATASET_DIR),
        "device": device,
        "selected_cases": [asdict(case) for case in SELECTED_CASES],
        "results": [asdict(result) for result in results],
        "summary": {
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Selected system tests")
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Device: {report['device']}")
    print("")
    for result in report["results"]:
        print(f"[{result['status'].upper()}] {result['case_id']} - {result['title']}")
        print(f"  File: {result['audio_name']}")
        print(f"  Function: {result['function_under_test']}")
        print(f"  Elapsed: {result['elapsed_sec']}s")
        for metric_name, metric_value in result["metrics"].items():
            print(f"  {metric_name}: {metric_value}")
        for note in result["notes"]:
            print(f"  note: {note}")
        print("")

    summary = report["summary"]
    print(f"Summary: {summary['passed']} passed, {summary['failed']} failed, {summary['total_cases']} total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three focused system tests on the prepared Vietnamese benchmark data.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ASR device to use.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "selected_system_tests_2026-04-10.json"),
        help="Path to the JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = PhoWhisperAdapter(device=args.device)
    results = [
        run_short_clean_asr_case(SELECTED_CASES[0], adapter),
        run_vad_preprocessing_case(SELECTED_CASES[1]),
        run_longform_chunking_case(SELECTED_CASES[2], adapter),
    ]

    report = build_report(results, device=args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"JSON report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
