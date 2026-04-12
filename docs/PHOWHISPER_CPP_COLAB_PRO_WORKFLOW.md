# PhoWhisper.cpp Colab Pro Workflow

Date: 2026-04-11

## Goal

This runbook pushes the heavy `PhoWhisper.cpp` experiment lane to Colab Pro so the local workstation only receives frozen outputs:

- converted `PhoWhisper.cpp` artifacts
- benchmark reports
- paper-ready figures
- qualitative failure evidence
- one final ZIP bundle for archival

## Main Artifact

Notebook:

- `notebooks/colab/PhoWhisper_CPP_Colab_Pro.ipynb`

CLI toolkit:

- `scripts/phowhisper_cpp_experiment.py`

## What the notebook does

1. Installs Colab-side dependencies only.
2. Loads this repo or a prebuilt toolkit bundle.
3. Downloads the benchmark dataset directly from a cloud URL or mounted Drive path.
4. Downloads `vinai/PhoWhisper-large`.
5. Clones and builds `whisper.cpp` plus `openai-whisper`.
6. Converts PhoWhisper into `whisper.cpp` runtime artifacts.
7. Optionally quantizes to `q4_0` and `q5_0`.
8. Benchmarks the converted models against Whisper baselines.
9. Renders figures and qualitative failure evidence.
10. Packages the full result set into one ZIP for download.

## Recommended Dataset Delivery

Preferred order:

1. Direct `https://...zip` URL
2. `gdrive://FILE_ID`
3. Mounted Drive path such as `/content/drive/MyDrive/...`

Avoid manual upload from local unless the archive is tiny. Large benchmark archives should be fetched inside Colab to save time and reduce browser failures.

## Evidence Produced For Paper

Expected outputs under `.../run_colab_canonical/evidence/`:

- `paper_evidence.md`
- `paper_evidence.json`
- `qualitative_cases.md`
- `figures/avg_wer_by_model.png`
- `figures/avg_cer_by_model.png`
- `figures/avg_rtf_by_model.png`
- `figures/avg_repeat_3gram_ratio_by_model.png`
- `figures/wer_vs_rtf_scatter.png`
- `figures/wer_boxplot_by_model.png`
- `figures/wer_heatmap.png`
- `figures/tail_risk_by_model.png`
- `figures/conversion_artifact_sizes.png`

These are the main paper clues:

- quality difference
- speed tradeoff
- quantization size tradeoff
- tail-risk behavior
- repetition or loop behavior as a hallucination proxy
- qualitative failure excerpts for discussion

## Download Checklist

Always download and freeze:

1. The final ZIP bundle from `package-results`
2. `conversion_manifest.json`
3. `benchmark_report.json`
4. `metrics_per_file.json`
5. `metrics_summary.json`
6. `paper_evidence.json`
7. All generated figures

## Notes

- The notebook is intentionally Colab-first and assumes internet access inside Colab.
- Cherry Core remains the local archival and analysis home.
- Sensitive datasets should only be run on Colab if policy permits external cloud execution.
