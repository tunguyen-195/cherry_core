from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import soundfile as sf

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.adapters.asr.stablets_adapter import StableTsAdapter
from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter

DATASET_DIR = Path("data/datasets/benchmark_vi_longform_v1")
SELECTED_FILES = {
    "audio_44": "longform_mixed_phrases",
    "audio_46": "longform_dense_transition",
    "audio_49": "longform_high_variety",
}


def try_set_low_priority() -> str:
    if os.name != "nt":
        return "default"
    try:
        import ctypes

        below_normal_priority_class = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.kernel32.SetPriorityClass(handle, below_normal_priority_class)
        return "below_normal" if ok else "default"
    except Exception:
        return "default"


def normalize_words(text: str) -> list[str]:
    text = (text or "").lower().replace("\n", " ")
    for token in ',.!?;:"“”()[]{}':
        text = text.replace(token, " ")
    return [word for word in text.split() if word]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / len(ref)


def summarize_transcript(transcript, audio_duration: float) -> dict[str, float | int | bool]:
    segments = transcript.segments or []
    word_count = sum(len(segment.get("words", [])) for segment in segments)
    segment_durations = [
        max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
        for segment in segments
    ]
    speech_coverage = sum(segment_durations)
    short_segments = sum(1 for duration in segment_durations if duration < 1.0)

    return {
        "segments": len(segments),
        "words_with_timestamps": word_count,
        "first_start": round(float(segments[0].get("start", 0.0)), 2) if segments else None,
        "last_end": round(float(segments[-1].get("end", 0.0)), 2) if segments else None,
        "speech_coverage_sec": round(speech_coverage, 2),
        "speech_coverage_ratio": round((speech_coverage / audio_duration), 4) if audio_duration else 0.0,
        "avg_segment_sec": round((speech_coverage / len(segments)), 2) if segments else 0.0,
        "short_segments_under_1s": short_segments,
        "text_chars": len(transcript.text or ""),
    }


def transcribe_with_timing(adapter, audio_path: Path):
    started = time.perf_counter()
    transcript = adapter.transcribe(str(audio_path))
    elapsed = time.perf_counter() - started
    return transcript, round(elapsed, 2)


def main() -> None:
    output_dir = Path("output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stable_ts_longform_small_{date.today().isoformat()}.json"
    priority_mode = try_set_low_priority()

    baseline = WhisperV2Adapter(device="cpu", use_vad=False, cpu_threads=1)
    stable_ts = StableTsAdapter(device="cpu", use_vad=False, cpu_threads=1)
    results = []

    for stem, purpose in SELECTED_FILES.items():
        audio_path = DATASET_DIR / f"{stem}.wav"
        transcript_path = DATASET_DIR / f"{stem}.txt"
        reference = transcript_path.read_text(encoding="utf-8").strip()
        audio_duration = sf.info(str(audio_path)).duration

        baseline_result, baseline_sec = transcribe_with_timing(baseline, audio_path)
        stable_result, stable_sec = transcribe_with_timing(stable_ts, audio_path)

        baseline_summary = summarize_transcript(baseline_result, audio_duration)
        stable_summary = summarize_transcript(stable_result, audio_duration)

        results.append(
            {
                "file": stem,
                "purpose": purpose,
                "audio_duration_sec": round(audio_duration, 2),
                "reference_words": len(normalize_words(reference)),
                "baseline_runtime_sec": baseline_sec,
                "stable_ts_runtime_sec": stable_sec,
                "baseline_wer": round(wer(reference, baseline_result.text), 4),
                "stable_ts_wer": round(wer(reference, stable_result.text), 4),
                "baseline": baseline_summary,
                "stable_ts": stable_summary,
                "text_changed": baseline_result.text != stable_result.text,
                "segment_count_changed": baseline_summary["segments"] != stable_summary["segments"],
                "short_segment_delta": stable_summary["short_segments_under_1s"] - baseline_summary["short_segments_under_1s"],
            }
        )

    summary = {
        "dataset": str(DATASET_DIR),
        "device": "cpu",
        "cpu_threads": 1,
        "priority_mode": priority_mode,
        "benchmark_type": "offline_longform_small",
        "files": list(SELECTED_FILES.keys()),
        "results": results,
        "average_baseline_wer": round(sum(item["baseline_wer"] for item in results) / len(results), 4),
        "average_stable_ts_wer": round(sum(item["stable_ts_wer"] for item in results) / len(results), 4),
        "average_baseline_runtime_sec": round(sum(item["baseline_runtime_sec"] for item in results) / len(results), 2),
        "average_stable_ts_runtime_sec": round(sum(item["stable_ts_runtime_sec"] for item in results) / len(results), 2),
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
