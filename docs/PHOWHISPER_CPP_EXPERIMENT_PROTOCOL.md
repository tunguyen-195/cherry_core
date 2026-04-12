# PhoWhisper.cpp Experiment Protocol

Date: 2026-04-11

## Purpose

This protocol turns the unfinished PhoWhisper `whisper.cpp` research from the older workspace into a reusable experiment lane inside Cherry Core.

The target is not to claim a new trained architecture.

The target is to claim a **reproducible Vietnamese ASR systems contribution**:

1. Convert a Vietnamese fine-tuned Whisper checkpoint into a C/C++ runtime artifact.
2. Quantize it for CPU deployment.
3. Benchmark accuracy, runtime, and hallucination proxy against baseline Whisper models.

That is a stronger and more defensible paper angle than simply saying "we used PhoWhisper".

## Local vs Colab Pro

Short recommendation:

- use **Colab Pro** for the heavy reproducible run
- use **local** for result review, report writing, and archival packaging

Reasoning:

### Why Colab Pro is better for the canonical experiment

1. Cleaner paper story
   - A Colab notebook/script is easier to package as a reproducible supplement.
   - Reviewers understand a cloud notebook workflow more easily than a machine-specific Windows setup.

2. Lower risk of interfering with your main workstation
   - Your local machine is already running other important lab workloads.
   - Conversion, quantization, and multi-model benchmark runs can be long and noisy.

3. Easier dependency reset
   - If a converter or `whisper.cpp` build breaks, restarting a Colab runtime is cleaner than repairing a local Windows environment.

4. Better separation between “research runtime” and “product runtime”
   - Cherry Core remains stable locally.
   - The experimental `PhoWhisper.cpp` lane can iterate independently in Colab.

### Why local is still important

1. Better control of evidence retention
   - Long-term storage is safer locally than in an ephemeral notebook VM.

2. Better visualization review
   - Tables, plots, transcripts, and failure cases are easier to inspect and curate locally.

3. Better final packaging
   - Canonical archives, ZIP bundles, and paper figures should be frozen on local disk after each Colab run.

### What this means in practice

Recommended split:

1. Run conversion + benchmark in Colab Pro.
2. Download all raw outputs immediately after each completed run.
3. Promote one run to a local canonical run directory.
4. Generate the final paper figures and tables locally from the downloaded artifacts.

If you only choose one environment:

- choose **Colab Pro** for the main experimental run
- choose **local** only for small sanity checks and report generation

## Suggested novelty angle for the paper

Safe wording:

- A reproducible pipeline for converting and evaluating a Vietnamese fine-tuned Whisper model inside a local C/C++ runtime.
- A study of quantization sensitivity for a Vietnamese Whisper derivative under CPU-only inference.
- An evaluation of transcript stability and hallucination proxy after moving from Python/Hugging Face runtime to `whisper.cpp`.

Claims to avoid unless you have direct evidence:

- "first ever PhoWhisper.cpp"
- "official PhoWhisper C++ release"
- "universally better on all Vietnamese tasks"

## What was added in Cherry Core

Main script:

- `scripts/phowhisper_cpp_experiment.py`

Reusable helpers:

- `research/phowhisper_cpp/workspace.py`
- `research/phowhisper_cpp/benchmarking.py`
- `research/phowhisper_cpp/transfer.py`
- `research/phowhisper_cpp/evidence.py`

Research notes:

- `docs/PHOWHISPER_CPP_RESEARCH_2026-04-11.md`
- `docs/PHOWHISPER_CPP_EXPERIMENT_PROTOCOL.md`
- `docs/PHOWHISPER_CPP_COLAB_PRO_WORKFLOW.md`
- `notebooks/colab/PhoWhisper_CPP_Colab_Pro.ipynb`

## Core commands

### 1. Initialize a clean experiment workspace

```powershell
python scripts/phowhisper_cpp_experiment.py init-workspace
```

Default output:

- `output/phowhisper_cpp_workspace`

### 2. Convert PhoWhisper to whisper.cpp runtime artifacts

```powershell
python scripts/phowhisper_cpp_experiment.py convert `
  --workspace-root output/phowhisper_cpp_workspace `
  --whisper-cpp-dir D:\path\to\whisper.cpp `
  --openai-whisper-dir D:\path\to\openai-whisper `
  --hf-model-dir E:\research\Cherry2\cherry_core\models\phowhisper-safe `
  --q4-mode q4_0 `
  --q5-mode q5_0
```

This step:

1. Ensures `model.safetensors` exists if only `pytorch_model.bin` is present.
2. Finds the available `whisper.cpp` converter script.
3. Runs conversion to a runtime artifact.
4. Quantizes to `q4_0` and `q5_0` unless disabled.
5. Writes `conversion_manifest.json`.

### 3. Benchmark the converted model

```powershell
python scripts/phowhisper_cpp_experiment.py benchmark `
  --dataset-dir E:\Freelance\Research\D12_02.2026_NCKH2026\ASR\data\datasets\benchmark_vi_longform_v1 `
  --output-dir output\phowhisper_cpp_workspace\artifacts\benchmarks\whispercpp\q5_40_long\run_local `
  --whisper-cli D:\path\to\whisper.cpp\build\bin\whisper-cli.exe `
  --language vi `
  --threads 8 `
  --model pho_large_q5=E:\path\to\ggml-phowhisper-large-q5_0.bin `
  --model whisper_large_v2_q5=E:\path\to\ggml-large-v2-q5_0.bin `
  --model whisper_large_v3_q5=E:\path\to\ggml-large-v3-q5_0.bin
```

Outputs include:

- `dataset_summary.json`
- `metrics_per_file.json`
- `metrics_summary.json`
- `benchmark_report.json`
- `benchmark_report.md`

### 3.5. Fetch a dataset directly into the workspace

From a cloud URL:

```powershell
python scripts/phowhisper_cpp_experiment.py fetch-dataset `
  --workspace-root output/phowhisper_cpp_workspace `
  --dataset-id benchmark_vi_longform_v1 `
  --source-url https://example.com/benchmark_vi_longform_v1.zip `
  --force
```

From a Google Drive file id:

```powershell
python scripts/phowhisper_cpp_experiment.py fetch-dataset `
  --workspace-root output/phowhisper_cpp_workspace `
  --dataset-id benchmark_vi_longform_v1 `
  --source-url gdrive://YOUR_FILE_ID `
  --force
```

### 3.6. Render paper evidence after benchmark

```powershell
python scripts/phowhisper_cpp_experiment.py render-evidence `
  --benchmark-dir output\phowhisper_cpp_workspace\artifacts\benchmarks\whispercpp\q5_40_long\run_local `
  --conversion-dir output\phowhisper_cpp_workspace\artifacts\conversion\conversion_2026-04-11_000000Z
```

This step creates:

- benchmark figures
- qualitative failure cases
- paper evidence markdown and JSON summaries

### 3.7. Package benchmark + evidence + converted models

```powershell
python scripts/phowhisper_cpp_experiment.py package-results `
  --benchmark-dir output\phowhisper_cpp_workspace\artifacts\benchmarks\whispercpp\q5_40_long\run_local `
  --conversion-dir output\phowhisper_cpp_workspace\artifacts\conversion\conversion_2026-04-11_000000Z `
  --output-zip output\phowhisper_cpp_workspace\artifacts\benchmarks\whispercpp\q5_40_long\paper_bundle.zip
```

### 4. Package the toolkit for Colab

```powershell
python scripts/phowhisper_cpp_experiment.py package-colab
```

Default output:

- `output/phowhisper_cpp_colab_bundle.zip`

## Minimum experimental matrix

Recommended matrix for the paper:

1. `PhoWhisper-large` Python/HF baseline
2. `PhoWhisper-large` `whisper.cpp` FP16
3. `PhoWhisper-large` `whisper.cpp` `q4_0`
4. `PhoWhisper-large` `whisper.cpp` `q5_0`
5. `Whisper large-v2` `whisper.cpp` `q5_0`
6. `Whisper large-v3` `whisper.cpp` `q5_0`

If compute budget is tight:

1. Keep Python/HF baseline on a smaller slice.
2. Run the full long-form slice only for the `whisper.cpp` models.

## Metrics to report

Mandatory:

- WER
- CER
- average RTF
- median RTF
- model file size

Recommended:

- worst-file case analysis
- number of files with `WER > 0.3`
- number of files with `WER > 0.5`
- `hyp/ref` word ratio as a hallucination proxy

## Evidence retention policy

For a paper, do not keep only the final markdown report.

Keep all of the following:

1. Conversion artifacts
   - `conversion_manifest.json`
   - SHA256 hashes
   - exact quantization mode used
   - exact `whisper.cpp` commit or archive

2. Benchmark outputs
   - `dataset_summary.json`
   - `metrics_per_file.json`
   - `metrics_summary.json`
   - `benchmark_report.json`
   - `benchmark_report.md`
   - raw prediction `.txt` files per model per audio
   - `stdout` and `stderr` logs for failed or suspicious files

3. Visual evidence
   - accuracy-speed scatter plot
   - average WER / CER / RTF bar charts
   - WER heatmap by file and model
   - worst-case transcript comparison figures

4. Run provenance
   - notebook or script version
   - dataset manifest
   - model file hashes
   - runtime environment notes
   - execution timestamp in UTC

Recommended archive unit:

- one ZIP per run
- one canonical promoted run directory
- one short summary table for the paper draft

## Visualization strategy for the paper

Best figure set:

1. Main tradeoff figure
   - x-axis: average RTF
   - y-axis: average WER
   - points: Python baseline, `whisper.cpp` FP16, `q4_0`, `q5_0`

2. Main comparison table
   - model
   - runtime
   - precision
   - Avg WER
   - Avg CER
   - Avg RTF
   - tail risk counts

3. Worst-case qualitative figure
   - reference transcript
   - PhoWhisper Python output
   - PhoWhisper.cpp output
   - one baseline Whisper output

4. Hallucination proxy figure
   - per-model count of high-WER files
   - optional `hyp/ref` word ratio distribution

Why this set is good:

- one figure tells the systems story
- one table gives quantitative credibility
- one qualitative figure demonstrates real error behavior
- one hallucination-oriented figure supports your paper narrative beyond WER

## Why this is useful for the paper

This experiment can support a stronger contribution claim if framed correctly:

- It studies what happens when a Vietnamese fine-tuned Whisper model is moved from a research runtime to a deployment runtime.
- It isolates conversion loss, quantization loss, and CPU deployment behavior.
- It directly serves the offline, local, operational use case.

That is more publication-friendly than a generic "we used a Vietnamese ASR model" statement.

## Known risks

- `whisper.cpp` converter behavior may differ by commit.
- Some converter paths still assume legacy `ggml` workflows.
- `q5_0` is the safest default unless your build clearly supports another Whisper-specific Q5 variant.
- Long-form synthetic benchmark results must not be over-generalized to all natural Vietnamese speech.

## Practical recommendation

For paper novelty:

- keep the `PhoWhisper.cpp` lane
- but also run a `PhoWhisper-ct2` control lane if possible

Reason:

- If `whisper.cpp` conversion shows accuracy regression, the CT2 lane becomes the engineering control.
- If `whisper.cpp` stays close in accuracy and gains deployability, that becomes a stronger publication result.
