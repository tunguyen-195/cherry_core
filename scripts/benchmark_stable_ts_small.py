from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.adapters.asr.stablets_adapter import StableTsAdapter
from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter

DATASET_DIR = Path("data/datasets/benchmark_vi_longform_v1")
SELECTED_FILES = {
    "audio_04": "clean_short_accuracy",
    "audio_07": "colloquial_short_accuracy",
    "audio_02": "formal_short_accuracy",
}


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


def main() -> None:
    output_dir = Path("output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stable_ts_small_{date.today().isoformat()}.json"

    baseline = WhisperV2Adapter(device="cpu", use_vad=False)
    stable_ts = StableTsAdapter(device="cpu", use_vad=False)
    results = []

    for stem, purpose in SELECTED_FILES.items():
        audio_path = DATASET_DIR / f"{stem}.wav"
        transcript_path = DATASET_DIR / f"{stem}.txt"
        reference = transcript_path.read_text(encoding="utf-8").strip()

        baseline_result = baseline.transcribe(str(audio_path))
        stable_result = stable_ts.transcribe(str(audio_path))

        baseline_wer = round(wer(reference, baseline_result.text), 4)
        stable_wer = round(wer(reference, stable_result.text), 4)

        results.append(
            {
                "file": stem,
                "purpose": purpose,
                "reference": reference,
                "baseline_text": baseline_result.text,
                "stable_ts_text": stable_result.text,
                "baseline_wer": baseline_wer,
                "stable_ts_wer": stable_wer,
                "delta_wer": round(stable_wer - baseline_wer, 4),
                "baseline_segments": len(baseline_result.segments),
                "stable_ts_segments": len(stable_result.segments),
                "stable_ts_changed_text": baseline_result.text != stable_result.text,
            }
        )

    summary = {
        "dataset": str(DATASET_DIR),
        "device": "cpu",
        "files": list(SELECTED_FILES.keys()),
        "results": results,
        "average_baseline_wer": round(sum(item["baseline_wer"] for item in results) / len(results), 4),
        "average_stable_ts_wer": round(sum(item["stable_ts_wer"] for item in results) / len(results), 4),
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
