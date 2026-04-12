# Evidence Registry

Date: 2026-04-11

## Purpose

This registry maps paper-usable claims to concrete repository artifacts.

Use only the artifacts below for tables, figures, and result statements.

## Repository Code Evidence

| Area | Artifact | What it supports |
| --- | --- | --- |
| Pipeline orchestration | `application/services/stt_web_pipeline.py` | layered optional processing flow, step ordering, artifact production |
| Conservative Whisper runtime | `infrastructure/adapters/asr/whisperv2_adapter.py` | `faster-whisper` backend, decode guardrails, repetition cleanup, offline local model loading |
| Stable-TS option | `infrastructure/adapters/asr/stablets_adapter.py` | optional offline stabilization path using `stable_whisper` with local `faster-whisper` model |
| Hallucination post-filter | `infrastructure/adapters/asr/hallucination_filter.py` | BoH and delooping implementation |
| VAD preprocessing | `infrastructure/adapters/vad/silero_adapter.py` | silence removal and conservative speech-preservation strategy |
| LLM correction | `application/services/correction_service.py` | downstream correction path, not core paper claim |
| Investigative analysis | `application/services/analysis_service.py`, `prompts/templates/deep_investigation.j2` | downstream structured intelligence application only |
| PhoWhisper.cpp toolkit | `scripts/phowhisper_cpp_experiment.py`, `research/phowhisper_cpp/*` | reproducible conversion, benchmark packaging, evidence rendering |

## Local Benchmark Evidence

### Selected system tests

Artifact:

- `output/benchmarks/selected_system_tests_2026-04-10_final.json`

Supports:

- short clean ASR sanity check
- VAD execution availability on the local offline path
- long-form chunk-stitching behavior on one selected case

Paper-safe use:

- as a targeted system-validation artifact
- not as a broad benchmark

Key numbers:

- `audio_01.wav`: `WER 0.0`, `CER 0.0`
- `audio_09.wav`: VAD executed successfully with duration reduction from `5.38s` to `3.69s`
- `audio_20.wav`: `WER 0.1`, repeated 8-gram hits `0`

### Local 3-file PhoWhisper benchmark

Artifact:

- `output/benchmark_vi_longform_sample3.json`

Supports:

- limited long-form Vietnamese accuracy/runtime snapshot for the current repo

Paper-safe use:

- only with explicit disclosure that this is a 3-file slice

Key numbers:

- files tested: `3`
- average `WER`: `0.1969`
- average `CER`: `0.1929`
- average `RTF`: `0.139`

## Pilot/Internal Evidence Imported from Older Research

### D12 PhoWhisper.cpp q5 benchmark

Artifact:

- `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/artifacts/benchmarks/whispercpp/q5_40_long/run_canonical/benchmark_report.json`

Supporting files:

- `benchmark_report.md`
- plots under `.../plots/`
- notebooks under `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/notebooks/colab/2026-03-18_phowhisper-large-whispercpp/`

Supports:

- pilot evidence that a converted and quantized `PhoWhisper` runtime can remain competitive on Vietnamese long-form audio
- early figure design for the paper

Required caveats:

- dataset marked `synthetic_longform_dataset`
- partial source metadata
- narrow duration band
- currently external to the main Cherry Core benchmark lane

Do not use as a main result without an in-repo confirmatory rerun.

## External Literature Evidence

| Source | URL | Use in paper |
| --- | --- | --- |
| PhoWhisper paper | https://arxiv.org/abs/2406.02555 | baseline Vietnamese ASR context |
| PhoWhisper repository | https://github.com/VinAIResearch/PhoWhisper | official implementation and benchmark claims |
| faster-whisper | https://github.com/SYSTRAN/faster-whisper | backend capability and fine-tuned model conversion/runtime context |
| whisper.cpp | https://github.com/ggml-org/whisper.cpp | C/C++ runtime context |
| Whisper hallucination study | https://arxiv.org/abs/2501.11378 | motivation for VAD and BoH-style mitigation |
| Calm-Whisper | https://arxiv.org/abs/2505.12969 | additional hallucination mitigation literature |

## Figure Provenance Rules

- Every figure must cite either a local artifact path or an external source URL.
- Any figure derived from D12 artifacts must be labeled `pilot/internal`.
- Any number without a source path should be removed from the manuscript draft.
