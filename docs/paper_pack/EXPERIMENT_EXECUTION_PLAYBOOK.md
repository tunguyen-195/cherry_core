# Experiment Execution Playbook

Date: 2026-04-11

## Purpose

This playbook turns the paper storyline into concrete, low-cost experiments that can be executed with minimal disruption to the main workstation.

Primary goals:

1. strengthen the main ASR mitigation claim
2. upgrade `PhoWhisper.cpp` from pilot evidence to in-repo confirmatory evidence
3. preserve all outputs in a paper-usable form

## Ground Rules

- Prefer small curated runs before any larger benchmark.
- Keep all artifacts under Cherry Core outputs.
- Do not claim any result that does not have a saved JSON or Markdown source.
- Keep GPU as the preferred product runtime, but isolate research runs so they do not interfere with other active work.
- For heavy `PhoWhisper.cpp` conversion and benchmark work, prefer Colab Pro and download the frozen artifacts back into the repo workspace.

## Experiment 1: Layered Mitigation Ablation

### Objective

Measure whether the reliability-oriented layers reduce practical failure modes on a small Vietnamese test set.

### Suggested file mix

Use 5 files if available:

- 2 short clean speech files
- 1 short noisy or hesitation-heavy file
- 2 long-form files

If resources are tight, run 3 files:

- 1 short clean
- 1 VAD-sensitive
- 1 long-form

### Conditions

Record outputs for each file under these conditions:

1. PhoWhisper baseline
2. PhoWhisper + VAD
3. whisper-v2 conservative path
4. whisper-v2 + Stable-TS
5. selected output + BoH/delooping

### Metrics

- `WER`
- `CER`
- repeated 3-gram hits
- repeated 8-gram hits
- word count ratio
- short qualitative note on obvious failure type

### Paper output

Create:

- one compact table for aggregate metrics
- one qualitative table with 3 to 5 representative errors

## Experiment 2: Long-Form Stability Review

### Objective

Analyze chunk-stitching behavior and false repetition in long recordings.

### Minimum setup

Run at least 3 long-form files.

### Checks

- overlap duplication
- repeated loops
- under-length hypothesis relative to reference
- sentence fragmentation
- silence-induced false phrases

### Paper output

Create:

- one table with `WER`, `CER`, repeated n-gram counts, and word-count ratio
- one short figure or heatmap if enough files are available

## Experiment 3: PhoWhisper.cpp Confirmatory Rerun

### Objective

Replace or reinforce the current pilot benchmark with a Cherry Core-managed evidence chain.

### Environment choice

Preferred:

- Colab Pro for conversion and heavy benchmark execution

Local usage:

- artifact review
- figure curation
- archival packaging

### Required outputs

- `conversion_manifest.json`
- `benchmark_report.json`
- `metrics_per_file.json`
- `metrics_summary.json`
- figure images
- one ZIP bundle containing all outputs

### Minimum comparison

Benchmark at least:

- `PhoWhisper.cpp` full precision or highest practical baseline
- `PhoWhisper.cpp` quantized variant
- `whisper-large-v2` control
- `whisper-large-v3` control

### Mandatory caveat handling

If the rerun uses a synthetic or concatenated dataset, label it clearly.
If the rerun uses a manually curated real slice, document the provenance in the report header.

## Artifact Discipline

For every experiment, save:

- raw prediction text
- scoring JSON
- short Markdown summary
- hardware note
- model configuration note

If a figure is created, store the source metrics next to it.

## Promotion Rules

Promote a result from `Pilot/Internal` to `Verified` only if:

1. the result is stored under Cherry Core
2. the dataset provenance is documented
3. the model/runtime configuration is recorded
4. the numbers can be re-derived from saved outputs

## Recommended Writing Sequence

1. Keep `MANUSCRIPT_DRAFT.md` as the main text.
2. Fill tables from experiment outputs only.
3. Update `CLAIM_MATRIX.md` after each completed benchmark pass.
4. Update `EVIDENCE_REGISTRY.md` when new artifacts are created.
