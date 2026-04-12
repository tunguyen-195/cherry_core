from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path

from research.phowhisper_cpp.benchmarking import (
    build_dataset_report,
    collect_dataset_entries,
    compute_cer,
    compute_repeat_ngram_ratio,
    compute_wer,
    normalize_text,
)
from research.phowhisper_cpp.evidence import render_paper_evidence
from research.phowhisper_cpp.transfer import materialize_dataset, package_artifacts_bundle
from research.phowhisper_cpp.workspace import load_project_paths_config, write_default_project_paths


def write_silence_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 0.25) -> None:
    frame_count = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def test_text_normalization_and_metrics_keep_vietnamese_characters():
    reference = "Hệ thống trinh sát âm thanh"
    hypothesis = "he thong  trinh sat am thanh!!!"

    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    assert normalized_reference == "hệ thống trinh sát âm thanh"
    assert normalized_hypothesis == "he thong trinh sat am thanh"
    assert compute_wer(normalized_reference, normalized_reference) == 0.0
    assert compute_cer(normalized_reference, normalized_reference) == 0.0


def test_workspace_config_bootstrap(tmp_path: Path):
    config_path = write_default_project_paths(tmp_path)
    config = load_project_paths_config(tmp_path)

    assert config_path.exists()
    assert config["paths"]["whisper_cpp_vendor"] == "external/vendor/whisper.cpp"
    assert (tmp_path / "external" / "models" / "ggml").exists()


def test_collect_dataset_entries_and_report(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    write_silence_wav(dataset_dir / "audio_10.wav", duration_sec=0.3)
    (dataset_dir / "audio_10.txt").write_text("Xin chào thế giới", encoding="utf-8")
    (dataset_dir / "audio_10.meta.txt").write_text(
        "\n".join(
            [
                "source_type=synthetic_longform",
                "source_dataset=demo",
                "generation_method=concatenated_streaming_samples",
                "source_sample_count=4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = collect_dataset_entries(dataset_dir, 10, 10)
    report, rows = build_dataset_report(entries, dataset_dir, 10, 10)

    assert len(entries) == 1
    assert rows[0]["audio_id"] == "audio_10"
    assert report["file_count"] == 1
    assert "synthetic_longform_dataset" in report["risk_flags"]


def test_repeat_ngram_ratio_detects_looping_text():
    clean_text = "day la mot cau binh thuong khong bi lap lai"
    loop_text = "mot cum lap lai mot cum lap lai mot cum lap lai"

    assert compute_repeat_ngram_ratio(clean_text, n=3) == 0.0
    assert compute_repeat_ngram_ratio(loop_text, n=3) > 0.5


def test_materialize_dataset_from_local_archive(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    write_default_project_paths(workspace_root)

    source_dataset = tmp_path / "source_dataset"
    source_dataset.mkdir()
    write_silence_wav(source_dataset / "audio_10.wav", duration_sec=0.2)
    (source_dataset / "audio_10.txt").write_text("xin chao", encoding="utf-8")

    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(source_dataset / "audio_10.wav", arcname="benchmark_vi_longform_v1/audio_10.wav")
        archive.write(source_dataset / "audio_10.txt", arcname="benchmark_vi_longform_v1/audio_10.txt")

    dataset_dir, manifest_path = materialize_dataset(
        workspace_root=workspace_root,
        dataset_id="benchmark_vi_longform_v1",
        archive_path=archive_path,
        force=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert dataset_dir.exists()
    assert (dataset_dir / "audio_10.wav").exists()
    assert manifest["file_counts"]["wav"] == 1


def test_render_paper_evidence_and_bundle(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark_run"
    benchmark_dir.mkdir()

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    transcript_path = dataset_root / "audio_10.txt"
    transcript_path.write_text("xin chao toi ten la an", encoding="utf-8")
    write_silence_wav(dataset_root / "audio_10.wav", duration_sec=0.2)

    prediction_dir = benchmark_dir / "predictions" / "pho_large_q5"
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / "audio_10.txt"
    prediction_path.write_text("xin chao toi ten la an", encoding="utf-8")

    dataset_rows = [
        {
            "audio_id": "audio_10",
            "file": "audio_10.wav",
            "duration_sec": 0.2,
            "reference_words": 6,
            "reference_chars": 22,
            "wav_path": str(dataset_root / "audio_10.wav"),
            "transcript_path": str(transcript_path),
            "meta_path": "",
            "metadata_status": "missing",
            "source_type": "unknown",
            "source_dataset": "unknown",
            "generation_method": "unknown",
            "source_sample_count": None,
        }
    ]
    benchmark_rows = [
        {
            "model": "pho_large_q5",
            "model_path": "fake_model.bin",
            "audio_id": "audio_10",
            "file": "audio_10.wav",
            "status": "report_only",
            "prediction_path": str(prediction_path),
            "stdout_path": str(prediction_path.with_suffix(".stdout.log")),
            "stderr_path": str(prediction_path.with_suffix(".stderr.log")),
            "duration_sec": 0.2,
            "runtime_sec": 0.05,
            "rtf": 0.25,
            "wer": 0.0,
            "cer": 0.0,
            "reference_word_count": 6,
            "hypothesis_word_count": 6,
            "word_ratio": 1.0,
            "repeat_3gram_ratio": 0.0,
            "max_3gram_repeat": 1,
            "delta_wer_vs_best": 0.0,
            "delta_cer_vs_best": 0.0,
            "delta_rtf_vs_fastest": 0.0,
        }
    ]
    model_summaries = [
        {
            "model": "pho_large_q5",
            "model_path": "fake_model.bin",
            "file_count": 1,
            "coverage": 1.0,
            "ok_count": 0,
            "reused_count": 0,
            "report_only_count": 1,
            "failed_count": 0,
            "timed_count": 1,
            "avg_wer": 0.0,
            "median_wer": 0.0,
            "avg_cer": 0.0,
            "median_cer": 0.0,
            "avg_rtf": 0.25,
            "median_rtf": 0.25,
            "avg_word_ratio": 1.0,
            "avg_repeat_3gram_ratio": 0.0,
            "max_3gram_repeat_observed": 1,
            "tail_wer_gt_03": 0,
            "tail_wer_gt_05": 0,
            "best_file": "audio_10",
            "worst_file": "audio_10",
            "rank_by_wer": 1,
            "rank_by_rtf": 1,
        }
    ]
    report_payload = {
        "generated_at": "2026-04-11T00:00:00+00:00",
        "config": {"dataset_dir": str(dataset_root), "language": "vi", "threads": 2, "report_only": True},
        "dataset_summary": {"file_count": 1, "risk_flags": []},
        "model_summaries": model_summaries,
        "hardest_files": [{"audio_id": "audio_10", "avg_wer": 0.0, "wer_gap": 0.0, "best_model": "pho_large_q5", "worst_model": "pho_large_q5"}],
    }

    (benchmark_dir / "dataset_per_file.json").write_text(json.dumps(dataset_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (benchmark_dir / "metrics_per_file.json").write_text(json.dumps(benchmark_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (benchmark_dir / "metrics_summary.json").write_text(json.dumps(model_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (benchmark_dir / "benchmark_report.json").write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    conversion_dir = tmp_path / "conversion"
    conversion_dir.mkdir()
    fake_model = conversion_dir / "ggml-phowhisper-large-q5_0.bin"
    fake_model.write_bytes(b"demo")
    (conversion_dir / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": fake_model.name,
                        "path": str(fake_model),
                        "size_bytes": fake_model.stat().st_size,
                        "sha256": "demo",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = render_paper_evidence(benchmark_dir=benchmark_dir, conversion_dir=conversion_dir, qualitative_top_n=1)
    bundle_path = package_artifacts_bundle(
        output_zip=tmp_path / "paper_bundle.zip",
        benchmark_dir=benchmark_dir,
        conversion_dir=conversion_dir,
    )

    assert outputs["paper_evidence_json"].exists()
    assert outputs["qualitative_cases"].exists()
    assert bundle_path.exists()
