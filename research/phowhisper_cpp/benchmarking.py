from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
import unicodedata
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"ok", "reused", "report_only"}


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_dir: Path
    output_dir: Path
    whisper_cli: Path
    language: str
    threads: int
    min_index: int
    max_index: int
    max_files: int | None
    models: list[tuple[str, Path]]
    extra_args: list[str]
    reuse_existing_output: bool
    report_only: bool


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    cleaned: list[str] = []
    pending_space = False
    for char in text:
        if char.isspace():
            pending_space = True
            continue
        if char.isalnum():
            if pending_space and cleaned:
                cleaned.append(" ")
            cleaned.append(char)
            pending_space = False
            continue
        pending_space = True
    return "".join(cleaned).strip()


def levenshtein(seq_a: list[Any], seq_b: list[Any]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    dp = list(range(len(seq_b) + 1))
    for index_a, item_a in enumerate(seq_a, start=1):
        previous = dp[0]
        dp[0] = index_a
        for index_b, item_b in enumerate(seq_b, start=1):
            cached = dp[index_b]
            cost = 0 if item_a == item_b else 1
            dp[index_b] = min(
                dp[index_b] + 1,
                dp[index_b - 1] + 1,
                previous + cost,
            )
            previous = cached
    return dp[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein(reference_words, hypothesis_words) / len(reference_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(list(reference), list(hypothesis)) / len(reference)


def compute_repeat_ngram_ratio(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated_instances = sum(count for count in counts.values() if count > 1)
    return repeated_instances / len(ngrams)


def compute_max_ngram_repeat(text: str, n: int = 3) -> int:
    tokens = text.split()
    if len(tokens) < n:
        return 1 if tokens else 0
    ngrams = [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    return max(counts.values(), default=0)


def audio_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_reader:
        return wav_reader.getnframes() / float(wav_reader.getframerate())


def collect_dataset_entries(
    dataset_dir: Path,
    min_index: int,
    max_index: int,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index in range(min_index, max_index + 1):
        audio_id = f"audio_{index:02d}"
        wav_path = dataset_dir / f"{audio_id}.wav"
        transcript_path = dataset_dir / f"{audio_id}.txt"
        meta_path = dataset_dir / f"{audio_id}.meta.txt"
        if not wav_path.exists() or not transcript_path.exists():
            continue

        reference_text = transcript_path.read_text(encoding="utf-8", errors="ignore").strip()
        entries.append(
            {
                "audio_id": audio_id,
                "file": wav_path.name,
                "wav_path": wav_path,
                "transcript_path": transcript_path,
                "meta_path": meta_path,
                "reference_text": reference_text,
                "reference_norm": normalize_text(reference_text),
                "duration_sec": audio_duration(wav_path),
                "metadata": parse_meta_file(meta_path),
            }
        )
        if max_files is not None and len(entries) >= max_files:
            break
    return entries


def parse_meta_file(meta_path: Path) -> dict[str, Any]:
    result = {
        "metadata_status": "missing",
        "source_type": "unknown",
        "source_dataset": "unknown",
        "generation_method": "unknown",
        "source_sample_count": None,
    }
    if not meta_path.exists():
        return result

    parsed: dict[str, str] = {}
    for raw_line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    result["metadata_status"] = "rich" if parsed else "legacy"
    result["source_type"] = parsed.get("source_type", "unknown")
    result["source_dataset"] = parsed.get("source_dataset", "unknown")
    result["generation_method"] = parsed.get("generation_method", "unknown")
    sample_count = parsed.get("source_sample_count")
    try:
        result["source_sample_count"] = int(sample_count) if sample_count else None
    except ValueError:
        result["source_sample_count"] = None
    return result


def summarize_numeric(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "avg": None, "median": None, "total": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(float(statistics.mean(values)), 6),
        "median": round(float(statistics.median(values)), 6),
        "total": round(float(sum(values)), 6),
    }


def build_dataset_report(
    entries: list[dict[str, Any]],
    dataset_dir: Path,
    min_index: int,
    max_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    durations = [entry["duration_sec"] for entry in entries]
    word_counts = [len(entry["reference_text"].split()) for entry in entries]
    char_counts = [len(entry["reference_text"]) for entry in entries]
    metadata_status_counts = Counter(entry["metadata"]["metadata_status"] for entry in entries)
    source_type_counts = Counter(entry["metadata"]["source_type"] for entry in entries)
    source_dataset_counts = Counter(entry["metadata"]["source_dataset"] for entry in entries)
    generation_method_counts = Counter(entry["metadata"]["generation_method"] for entry in entries)
    source_sample_counts = [
        int(entry["metadata"]["source_sample_count"])
        for entry in entries
        if entry["metadata"]["source_sample_count"] is not None
    ]

    report = {
        "dataset_dir": str(dataset_dir),
        "index_range": [min_index, max_index],
        "file_count": len(entries),
        "duration_sec": summarize_numeric(durations),
        "transcript_words": summarize_numeric(word_counts),
        "transcript_chars": summarize_numeric(char_counts),
        "source_sample_count": summarize_numeric(source_sample_counts),
        "metadata_status_counts": dict(metadata_status_counts),
        "source_type_counts": dict(source_type_counts),
        "source_dataset_counts": dict(source_dataset_counts),
        "generation_method_counts": dict(generation_method_counts),
        "risk_flags": [],
    }

    if source_type_counts.get("synthetic_longform", 0) > 0:
        report["risk_flags"].append("synthetic_longform_dataset")
    if metadata_status_counts.get("legacy", 0) > 0 or metadata_status_counts.get("missing", 0) > 0:
        report["risk_flags"].append("partial_source_metadata")
    if len(durations) > 1 and max(durations) - min(durations) < 10:
        report["risk_flags"].append("narrow_duration_band")

    per_file_rows = []
    for entry in entries:
        per_file_rows.append(
            {
                "audio_id": entry["audio_id"],
                "file": entry["file"],
                "duration_sec": round(entry["duration_sec"], 4),
                "reference_words": len(entry["reference_text"].split()),
                "reference_chars": len(entry["reference_text"]),
                "wav_path": str(entry["wav_path"]),
                "transcript_path": str(entry["transcript_path"]),
                "meta_path": str(entry["meta_path"]) if entry["meta_path"].exists() else "",
                **entry["metadata"],
            }
        )
    return report, per_file_rows


def run_model_on_file(
    cli_path: Path,
    model_path: Path,
    wav_path: Path,
    out_prefix: Path,
    language: str,
    threads: int,
    extra_args: list[str],
) -> tuple[int, float, str, str]:
    command = [
        str(cli_path),
        "-m",
        str(model_path),
        "-l",
        language,
        "-t",
        str(threads),
        "-f",
        str(wav_path),
        "-otxt",
        "-of",
        str(out_prefix),
        "-nt",
        "-np",
    ]
    command.extend(extra_args)

    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    runtime_sec = time.perf_counter() - started_at
    return completed.returncode, runtime_sec, completed.stdout, completed.stderr


def benchmark_models(entries: list[dict[str, Any]], config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for label, model_path in config.models:
        model_output_dir = config.output_dir / "predictions" / label
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            out_prefix = model_output_dir / entry["audio_id"]
            prediction_path = out_prefix.with_suffix(".txt")
            stdout_path = out_prefix.with_suffix(".stdout.log")
            stderr_path = out_prefix.with_suffix(".stderr.log")

            runtime_sec = None
            return_code = None
            status = "pending"

            if config.report_only:
                status = "report_only" if prediction_path.exists() else "missing_prediction"
            elif config.reuse_existing_output and prediction_path.exists():
                status = "reused"
            else:
                return_code, runtime_sec, stdout_text, stderr_text = run_model_on_file(
                    cli_path=config.whisper_cli,
                    model_path=model_path,
                    wav_path=entry["wav_path"],
                    out_prefix=out_prefix,
                    language=config.language,
                    threads=config.threads,
                    extra_args=config.extra_args,
                )
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")
                status = "ok" if return_code == 0 and prediction_path.exists() else "failed"

            prediction_text = prediction_path.read_text(encoding="utf-8", errors="ignore").strip() if prediction_path.exists() else ""
            prediction_norm = normalize_text(prediction_text)
            rtf = runtime_sec / entry["duration_sec"] if runtime_sec and entry["duration_sec"] > 0 else None
            success = status in SUCCESS_STATUSES and prediction_path.exists()

            rows.append(
                {
                    "model": label,
                    "model_path": str(model_path),
                    "audio_id": entry["audio_id"],
                    "file": entry["file"],
                    "status": status,
                    "prediction_path": str(prediction_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "duration_sec": round(entry["duration_sec"], 4),
                    "runtime_sec": round(runtime_sec, 4) if runtime_sec is not None else None,
                    "rtf": round(rtf, 6) if rtf is not None else None,
                    "wer": round(compute_wer(entry["reference_norm"], prediction_norm), 6) if success else None,
                    "cer": round(compute_cer(entry["reference_norm"], prediction_norm), 6) if success else None,
                    "reference_word_count": len(entry["reference_norm"].split()),
                    "hypothesis_word_count": len(prediction_norm.split()),
                    "word_ratio": round(
                        len(prediction_norm.split()) / max(1, len(entry["reference_norm"].split())),
                        6,
                    )
                    if success
                    else None,
                    "repeat_3gram_ratio": round(compute_repeat_ngram_ratio(prediction_norm, n=3), 6) if success else None,
                    "max_3gram_repeat": compute_max_ngram_repeat(prediction_norm, n=3) if success else None,
                }
            )

    attach_relative_metrics(rows)
    return rows


def attach_relative_metrics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["audio_id"]].append(row)

    for file_rows in grouped.values():
        valid_rows = [row for row in file_rows if row["wer"] is not None]
        timed_rows = [row for row in file_rows if row["rtf"] is not None]
        best_wer = min((row["wer"] for row in valid_rows), default=None)
        best_cer = min((row["cer"] for row in valid_rows), default=None)
        fastest_rtf = min((row["rtf"] for row in timed_rows), default=None)
        for row in file_rows:
            row["delta_wer_vs_best"] = round(row["wer"] - best_wer, 6) if row["wer"] is not None and best_wer is not None else None
            row["delta_cer_vs_best"] = round(row["cer"] - best_cer, 6) if row["cer"] is not None and best_cer is not None else None
            row["delta_rtf_vs_fastest"] = round(row["rtf"] - fastest_rtf, 6) if row["rtf"] is not None and fastest_rtf is not None else None


def safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def summarize_model_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    model_summaries: list[dict[str, Any]] = []
    for label, model_rows in grouped.items():
        success_rows = [row for row in model_rows if row["status"] in SUCCESS_STATUSES and row["wer"] is not None]
        timed_rows = [row for row in success_rows if row["rtf"] is not None]
        word_ratios = [row["word_ratio"] for row in success_rows if row["word_ratio"] is not None]
        repeat_ratios = [row["repeat_3gram_ratio"] for row in success_rows if row["repeat_3gram_ratio"] is not None]
        repeat_maxima = [row["max_3gram_repeat"] for row in success_rows if row["max_3gram_repeat"] is not None]
        model_summaries.append(
            {
                "model": label,
                "model_path": model_rows[0]["model_path"],
                "file_count": len(model_rows),
                "coverage": round(len(success_rows) / max(1, len(model_rows)), 6),
                "ok_count": sum(1 for row in model_rows if row["status"] == "ok"),
                "reused_count": sum(1 for row in model_rows if row["status"] == "reused"),
                "report_only_count": sum(1 for row in model_rows if row["status"] == "report_only"),
                "failed_count": sum(1 for row in model_rows if row["status"] == "failed"),
                "timed_count": len(timed_rows),
                "avg_wer": safe_round(statistics.mean([row["wer"] for row in success_rows])) if success_rows else None,
                "median_wer": safe_round(statistics.median([row["wer"] for row in success_rows])) if success_rows else None,
                "avg_cer": safe_round(statistics.mean([row["cer"] for row in success_rows])) if success_rows else None,
                "median_cer": safe_round(statistics.median([row["cer"] for row in success_rows])) if success_rows else None,
                "avg_rtf": safe_round(statistics.mean([row["rtf"] for row in timed_rows])) if timed_rows else None,
                "median_rtf": safe_round(statistics.median([row["rtf"] for row in timed_rows])) if timed_rows else None,
                "avg_word_ratio": safe_round(statistics.mean(word_ratios)) if word_ratios else None,
                "avg_repeat_3gram_ratio": safe_round(statistics.mean(repeat_ratios)) if repeat_ratios else None,
                "max_3gram_repeat_observed": max(repeat_maxima) if repeat_maxima else None,
                "tail_wer_gt_03": sum(1 for row in success_rows if row["wer"] is not None and row["wer"] > 0.3),
                "tail_wer_gt_05": sum(1 for row in success_rows if row["wer"] is not None and row["wer"] > 0.5),
                "best_file": min(success_rows, key=lambda row: row["wer"])["audio_id"] if success_rows else None,
                "worst_file": max(success_rows, key=lambda row: row["wer"])["audio_id"] if success_rows else None,
            }
        )

    wer_rank = sorted(
        model_summaries,
        key=lambda row: (
            float("inf") if row["avg_wer"] is None else row["avg_wer"],
            float("inf") if row["avg_rtf"] is None else row["avg_rtf"],
            row["model"],
        ),
    )
    for rank, summary in enumerate(wer_rank, start=1):
        summary["rank_by_wer"] = rank

    speed_rank = sorted(
        model_summaries,
        key=lambda row: (
            float("inf") if row["avg_rtf"] is None else row["avg_rtf"],
            float("inf") if row["avg_wer"] is None else row["avg_wer"],
            row["model"],
        ),
    )
    for rank, summary in enumerate(speed_rank, start=1):
        summary["rank_by_rtf"] = rank

    hardest_files: list[dict[str, Any]] = []
    by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_audio[row["audio_id"]].append(row)
    for audio_id, file_rows in by_audio.items():
        valid_rows = [row for row in file_rows if row["wer"] is not None]
        if not valid_rows:
            continue
        avg_wer = statistics.mean(row["wer"] for row in valid_rows)
        best_row = min(valid_rows, key=lambda row: row["wer"])
        worst_row = max(valid_rows, key=lambda row: row["wer"])
        hardest_files.append(
            {
                "audio_id": audio_id,
                "avg_wer": safe_round(avg_wer),
                "wer_gap": safe_round(worst_row["wer"] - best_row["wer"]),
                "best_model": best_row["model"],
                "worst_model": worst_row["model"],
            }
        )
    hardest_files.sort(key=lambda row: row["avg_wer"], reverse=True)
    return model_summaries, hardest_files


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown_report(report_payload: dict[str, Any]) -> str:
    config = report_payload["config"]
    dataset_summary = report_payload["dataset_summary"]
    model_summaries = report_payload["model_summaries"]
    hardest_files = report_payload["hardest_files"][:10]

    model_table = markdown_table(
        ["Rank", "Model", "Avg WER", "Avg CER", "Avg RTF", "Repeat-3g", "Tail >0.3", "Tail >0.5"],
        [
            [
                str(summary.get("rank_by_wer", "")),
                summary["model"],
                format_metric(summary["avg_wer"]),
                format_metric(summary["avg_cer"]),
                format_metric(summary["avg_rtf"]),
                format_metric(summary.get("avg_repeat_3gram_ratio")),
                str(summary["tail_wer_gt_03"]),
                str(summary["tail_wer_gt_05"]),
            ]
            for summary in sorted(model_summaries, key=lambda item: item.get("rank_by_wer", 999))
        ],
    )

    hardest_table = markdown_table(
        ["Audio", "Avg WER", "WER Gap", "Best", "Worst"],
        [
            [
                row["audio_id"],
                format_metric(row["avg_wer"]),
                format_metric(row["wer_gap"]),
                row["best_model"],
                row["worst_model"],
            ]
            for row in hardest_files
        ],
    )

    lines = [
        "# PhoWhisper.cpp Benchmark Report",
        "",
        f"- generated_at: `{report_payload['generated_at']}`",
        f"- dataset_dir: `{config['dataset_dir']}`",
        f"- language: `{config['language']}`",
        f"- threads: `{config['threads']}`",
        f"- report_only: `{config['report_only']}`",
        "",
        "## Dataset Summary",
        "",
        f"- file_count: `{dataset_summary['file_count']}`",
        f"- risk_flags: `{dataset_summary['risk_flags']}`",
        "",
        "## Model Summary",
        "",
        model_table,
        "",
        "## Hardest Files",
        "",
        hardest_table,
        "",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    config: BenchmarkConfig,
    dataset_report: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    hardest_files: list[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = output_dir / "dataset_summary.json"
    dataset_rows_path = output_dir / "dataset_per_file.json"
    metrics_path = output_dir / "metrics_per_file.json"
    summary_path = output_dir / "metrics_summary.json"
    report_json = output_dir / "benchmark_report.json"
    report_md = output_dir / "benchmark_report.md"
    metrics_csv = output_dir / "metrics_per_file.csv"

    dataset_summary_path.write_text(json.dumps(dataset_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_rows_path.write_text(json.dumps(dataset_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(benchmark_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(model_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "dataset_dir": str(config.dataset_dir),
            "output_dir": str(config.output_dir),
            "whisper_cli": str(config.whisper_cli),
            "language": config.language,
            "threads": config.threads,
            "min_index": config.min_index,
            "max_index": config.max_index,
            "max_files": config.max_files,
            "models": [{"label": label, "path": str(path)} for label, path in config.models],
            "extra_args": config.extra_args,
            "reuse_existing_output": config.reuse_existing_output,
            "report_only": config.report_only,
        },
        "dataset_summary": dataset_report,
        "model_summaries": model_summaries,
        "hardest_files": hardest_files,
    }
    report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmark_rows[0].keys()) if benchmark_rows else ["model"])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    report_md.write_text(render_markdown_report(report_payload), encoding="utf-8")

    return {
        "dataset_summary": dataset_summary_path,
        "dataset_rows": dataset_rows_path,
        "metrics_per_file": metrics_path,
        "metrics_csv": metrics_csv,
        "metrics_summary": summary_path,
        "benchmark_report_json": report_json,
        "benchmark_report_md": report_md,
    }
