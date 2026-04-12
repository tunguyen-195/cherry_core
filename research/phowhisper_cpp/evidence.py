from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workspace import write_json


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _excerpt(text: str, limit: int = 480) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _import_plotting() -> tuple[Any | None, Any | None, Exception | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency dependent
        return None, None, exc
    return plt, np, None


def _save_bar_chart(plt: Any, items: list[tuple[str, float]], title: str, ylabel: str, out_path: Path) -> None:
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#2E5B7C")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    for tick in axis.get_xticklabels():
        tick.set_rotation(15)
        tick.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _save_histogram(plt: Any, values: list[float], title: str, xlabel: str, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(values, bins=min(10, max(4, len(values) // 2 or 1)), color="#5F8F6F", edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _extract_top_repeated_3gram(text: str) -> tuple[str | None, int]:
    tokens = text.split()
    if len(tokens) < 3:
        return None, 0
    ngrams = [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    counts = Counter(ngrams)
    repeated = [(phrase, count) for phrase, count in counts.items() if count > 1]
    if not repeated:
        return None, 0
    repeated.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return repeated[0]


def render_benchmark_figures(
    dataset_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    plt, np, import_error = _import_plotting()
    if import_error is not None:
        note_path = figures_dir / "plotting_unavailable.txt"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            f"Plotting dependencies are unavailable in this environment: {import_error}\n",
            encoding="utf-8",
        )
        return [note_path]

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    durations = [float(row["duration_sec"]) for row in dataset_rows if row.get("duration_sec") is not None]
    if durations:
        path = figures_dir / "dataset_duration_hist.png"
        _save_histogram(plt, durations, "Dataset Duration Distribution", "Duration (seconds)", path)
        created.append(path)

    word_counts = [float(row["reference_words"]) for row in dataset_rows if row.get("reference_words") is not None]
    if word_counts:
        path = figures_dir / "dataset_word_count_hist.png"
        _save_histogram(plt, word_counts, "Transcript Word Count Distribution", "Words per file", path)
        created.append(path)

    for metric_key, title, ylabel, file_name in [
        ("avg_wer", "Average WER by Model", "WER", "avg_wer_by_model.png"),
        ("avg_cer", "Average CER by Model", "CER", "avg_cer_by_model.png"),
        ("avg_rtf", "Average RTF by Model", "RTF", "avg_rtf_by_model.png"),
        ("avg_repeat_3gram_ratio", "Average Repeated 3-gram Ratio by Model", "Repeat Ratio", "avg_repeat_3gram_ratio_by_model.png"),
    ]:
        items = [(row["model"], row[metric_key]) for row in model_summaries if row.get(metric_key) is not None]
        if not items:
            continue
        path = figures_dir / file_name
        _save_bar_chart(plt, items, title, ylabel, path)
        created.append(path)

    tail_rows = [
        (row["model"], float(row["tail_wer_gt_03"]), float(row["tail_wer_gt_05"]))
        for row in model_summaries
        if row.get("tail_wer_gt_03") is not None and row.get("tail_wer_gt_05") is not None
    ]
    if tail_rows:
        labels = [item[0] for item in tail_rows]
        over_03 = [item[1] for item in tail_rows]
        over_05 = [item[2] for item in tail_rows]
        positions = np.arange(len(labels))
        width = 0.36
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(positions - width / 2, over_03, width=width, label="WER > 0.3", color="#B85C38")
        axis.bar(positions + width / 2, over_05, width=width, label="WER > 0.5", color="#6C2E2E")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.set_ylabel("File count")
        axis.set_title("High-WER Tail Risk by Model")
        axis.legend()
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
        figure.tight_layout()
        path = figures_dir / "tail_risk_by_model.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    scatter_points = [row for row in model_summaries if row.get("avg_wer") is not None and row.get("avg_rtf") is not None]
    if scatter_points:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(
            [row["avg_rtf"] for row in scatter_points],
            [row["avg_wer"] for row in scatter_points],
            s=95,
            color="#8A4FFF",
        )
        for row in scatter_points:
            axis.annotate(row["model"], (row["avg_rtf"], row["avg_wer"]), textcoords="offset points", xytext=(6, 4))
        axis.set_title("Accuracy vs Speed Tradeoff")
        axis.set_xlabel("Average RTF (lower is faster)")
        axis.set_ylabel("Average WER (lower is better)")
        axis.grid(True, linestyle="--", alpha=0.35)
        figure.tight_layout()
        path = figures_dir / "wer_vs_rtf_scatter.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    box_groups: dict[str, list[float]] = defaultdict(list)
    for row in benchmark_rows:
        if row.get("wer") is not None:
            box_groups[row["model"]].append(float(row["wer"]))
    if box_groups:
        labels = list(box_groups.keys())
        values = [box_groups[label] for label in labels]
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.boxplot(values, tick_labels=labels, orientation="vertical")
        axis.set_title("Per-file WER Distribution by Model")
        axis.set_ylabel("WER")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        for tick in axis.get_xticklabels():
            tick.set_rotation(15)
            tick.set_horizontalalignment("right")
        figure.tight_layout()
        path = figures_dir / "wer_boxplot_by_model.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    heatmap_rows = [row for row in benchmark_rows if row.get("wer") is not None]
    if heatmap_rows:
        model_order = [row["model"] for row in model_summaries]
        audio_ids = sorted({row["audio_id"] for row in heatmap_rows})
        matrix = np.full((len(model_order), len(audio_ids)), np.nan)
        for row in heatmap_rows:
            matrix[model_order.index(row["model"]), audio_ids.index(row["audio_id"])] = row["wer"]
        figure, axis = plt.subplots(figsize=(max(12, len(audio_ids) * 0.45), max(4, len(model_order) * 0.9)))
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        axis.set_title("WER Heatmap by File and Model")
        axis.set_xticks(range(len(audio_ids)))
        axis.set_xticklabels(audio_ids, rotation=90)
        axis.set_yticks(range(len(model_order)))
        axis.set_yticklabels(model_order)
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("WER")
        figure.tight_layout()
        path = figures_dir / "wer_heatmap.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    return created


def render_conversion_figures(conversion_manifest: dict[str, Any], figures_dir: Path) -> list[Path]:
    plt, _, import_error = _import_plotting()
    if import_error is not None:
        note_path = figures_dir / "conversion_plotting_unavailable.txt"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            f"Conversion plotting dependencies are unavailable in this environment: {import_error}\n",
            encoding="utf-8",
        )
        return [note_path]

    files = conversion_manifest.get("files", [])
    if not files:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [item["name"] for item in files]
    sizes_mb = [round(item["size_bytes"] / (1024 * 1024), 3) for item in files]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, sizes_mb, color="#3F7D58")
    axis.set_title("PhoWhisper.cpp Artifact Sizes")
    axis.set_ylabel("Size (MB)")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    for tick in axis.get_xticklabels():
        tick.set_rotation(15)
        tick.set_horizontalalignment("right")
    figure.tight_layout()
    path = figures_dir / "conversion_artifact_sizes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [path]


def write_qualitative_cases(
    dataset_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    top_n: int = 3,
) -> Path:
    dataset_lookup = {row["audio_id"]: row for row in dataset_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        if row.get("wer") is not None:
            grouped[row["audio_id"]].append(row)

    ranked_audio_ids = sorted(
        grouped.keys(),
        key=lambda audio_id: sum(item["wer"] for item in grouped[audio_id]) / max(1, len(grouped[audio_id])),
        reverse=True,
    )

    lines = [
        "# Qualitative Failure Cases",
        "",
        "This file captures the hardest benchmark files together with transcript clues that are useful for paper discussion.",
        "",
    ]
    for audio_id in ranked_audio_ids[:top_n]:
        dataset_row = dataset_lookup.get(audio_id, {})
        transcript_path = Path(dataset_row.get("transcript_path", "")) if dataset_row.get("transcript_path") else None
        reference_text = transcript_path.read_text(encoding="utf-8", errors="ignore").strip() if transcript_path and transcript_path.exists() else ""
        lines.extend(
            [
                f"## {audio_id}",
                "",
                f"- file: `{dataset_row.get('file', audio_id)}`",
                f"- duration_sec: `{dataset_row.get('duration_sec', 'n/a')}`",
                f"- reference_excerpt: `{_excerpt(reference_text)}`",
                "",
            ]
        )
        for row in sorted(grouped[audio_id], key=lambda item: (item.get("wer", 1.0), item["model"])):
            prediction_path = Path(row["prediction_path"]) if row.get("prediction_path") else None
            hypothesis_text = prediction_path.read_text(encoding="utf-8", errors="ignore").strip() if prediction_path and prediction_path.exists() else ""
            top_repeat_phrase, top_repeat_count = _extract_top_repeated_3gram(" ".join(hypothesis_text.split()).lower())
            lines.extend(
                [
                    f"### {row['model']}",
                    "",
                    f"- wer: `{_format_metric(row.get('wer'))}`",
                    f"- cer: `{_format_metric(row.get('cer'))}`",
                    f"- rtf: `{_format_metric(row.get('rtf'))}`",
                    f"- word_ratio: `{_format_metric(row.get('word_ratio'))}`",
                    f"- repeat_3gram_ratio: `{_format_metric(row.get('repeat_3gram_ratio'))}`",
                    f"- max_3gram_repeat: `{row.get('max_3gram_repeat', 'n/a')}`",
                    f"- top_repeated_3gram: `{top_repeat_phrase or 'none'}` x `{top_repeat_count}`",
                    f"- hypothesis_excerpt: `{_excerpt(hypothesis_text)}`",
                    "",
                ]
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def render_paper_evidence(
    benchmark_dir: Path,
    *,
    conversion_dir: Path | None = None,
    qualitative_top_n: int = 3,
) -> dict[str, Path]:
    benchmark_dir = benchmark_dir.expanduser().resolve()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    dataset_rows = _load_json(benchmark_dir / "dataset_per_file.json")
    benchmark_rows = _load_json(benchmark_dir / "metrics_per_file.json")
    model_summaries = _load_json(benchmark_dir / "metrics_summary.json")
    report_payload = _load_json(benchmark_dir / "benchmark_report.json")

    evidence_dir = benchmark_dir / "evidence"
    figures_dir = evidence_dir / "figures"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    created_paths: dict[str, Path] = {}
    benchmark_figure_paths = render_benchmark_figures(dataset_rows, benchmark_rows, model_summaries, figures_dir)
    if benchmark_figure_paths:
        created_paths["figures_dir"] = figures_dir

    conversion_manifest = None
    conversion_figure_paths: list[Path] = []
    if conversion_dir is not None:
        conversion_dir = conversion_dir.expanduser().resolve()
        manifest_path = conversion_dir / "conversion_manifest.json"
        if manifest_path.exists():
            conversion_manifest = _load_json(manifest_path)
            conversion_figure_paths = render_conversion_figures(conversion_manifest, figures_dir)

    qualitative_cases_path = write_qualitative_cases(
        dataset_rows=dataset_rows,
        benchmark_rows=benchmark_rows,
        out_path=evidence_dir / "qualitative_cases.md",
        top_n=qualitative_top_n,
    )
    created_paths["qualitative_cases"] = qualitative_cases_path

    best_model = min(
        [row for row in model_summaries if row.get("avg_wer") is not None],
        key=lambda item: item["avg_wer"],
        default=None,
    )
    fastest_model = min(
        [row for row in model_summaries if row.get("avg_rtf") is not None],
        key=lambda item: item["avg_rtf"],
        default=None,
    )
    lowest_repeat_model = min(
        [row for row in model_summaries if row.get("avg_repeat_3gram_ratio") is not None],
        key=lambda item: item["avg_repeat_3gram_ratio"],
        default=None,
    )

    figure_records = [
        {"path": str(path), "name": path.name}
        for path in [*benchmark_figure_paths, *conversion_figure_paths]
    ]
    summary_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_dir": str(benchmark_dir),
        "conversion_dir": str(conversion_dir) if conversion_dir is not None else None,
        "dataset_risk_flags": report_payload.get("dataset_summary", {}).get("risk_flags", []),
        "best_model_by_wer": best_model,
        "fastest_model_by_rtf": fastest_model,
        "lowest_repeat_model": lowest_repeat_model,
        "figure_records": figure_records,
        "qualitative_cases_path": str(qualitative_cases_path),
    }
    summary_path = write_json(evidence_dir / "paper_evidence.json", summary_payload)
    created_paths["paper_evidence_json"] = summary_path

    markdown_lines = [
        "# Paper Evidence Pack",
        "",
        f"- generated_at: `{summary_payload['generated_at']}`",
        f"- benchmark_dir: `{benchmark_dir}`",
        f"- conversion_dir: `{conversion_dir}`" if conversion_dir is not None else "- conversion_dir: `n/a`",
        f"- dataset_risk_flags: `{summary_payload['dataset_risk_flags']}`",
        "",
        "## Topline Findings",
        "",
        f"- best_model_by_wer: `{best_model['model']}` with Avg WER `{_format_metric(best_model.get('avg_wer'))}`" if best_model else "- best_model_by_wer: `n/a`",
        f"- fastest_model_by_rtf: `{fastest_model['model']}` with Avg RTF `{_format_metric(fastest_model.get('avg_rtf'))}`" if fastest_model else "- fastest_model_by_rtf: `n/a`",
        f"- lowest_repeat_model: `{lowest_repeat_model['model']}` with Avg repeated 3-gram ratio `{_format_metric(lowest_repeat_model.get('avg_repeat_3gram_ratio'))}`"
        if lowest_repeat_model
        else "- lowest_repeat_model: `n/a`",
        "",
        "## Figure Inventory",
        "",
    ]
    if figure_records:
        markdown_lines.extend([f"- `{item['name']}`" for item in figure_records])
    else:
        markdown_lines.append("- No figures were generated.")
    markdown_lines.extend(
        [
            "",
            "## Qualitative Cases",
            "",
            f"- See `{qualitative_cases_path.name}` for transcript excerpts and repetition clues.",
            "",
        ]
    )
    markdown_path = evidence_dir / "paper_evidence.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    created_paths["paper_evidence_md"] = markdown_path
    return created_paths
