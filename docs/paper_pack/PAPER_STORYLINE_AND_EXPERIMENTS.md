# Paper Storyline and Experiments

Date: 2026-04-11

## Recommended Paper Direction

### Working title options

1. Cherry Core: Offline Vietnamese ASR with Layered Hallucination Mitigation and Local Deployment Support
2. Towards Reliable Offline Vietnamese ASR: A Layered Mitigation Pipeline with a PhoWhisper.cpp Runtime Study
3. Reliability-Oriented Vietnamese Speech Transcription: Anti-Hallucination Design and PhoWhisper.cpp Benchmarking

### Recommended title choice

Use title option 2.

It is technically accurate, avoids claiming model novelty, and leaves room for both the ASR mitigation stack and the `PhoWhisper.cpp` study.

## Contribution Framing

### Contribution 1

An offline Vietnamese ASR pipeline that combines:

- VAD preprocessing
- conservative Whisper-family decoding
- repetition cleanup
- optional Stable-TS stabilization
- optional BoH and delooping post-filtering

The contribution is the layered design and its empirical behavior on local Vietnamese test audio, not a new base model.

### Contribution 2

A reproducible path for evaluating `PhoWhisper.cpp` under local CPU deployment constraints:

- checkpoint conversion
- quantization
- benchmark packaging
- paper-ready evidence generation

This contribution is strongest when framed as deployment engineering and runtime evaluation.

### Application Section

Use the LLM investigative summary as an application demonstration only:

- structured intelligence extraction
- timeline and entity surfacing
- analyst-facing summarization

Do not make it part of the main scientific claim set unless a separate annotation protocol is added later.

## Recommended Paper Structure

### 1. Introduction

- motivate hallucination, repetition, and long-form instability in offline ASR
- motivate Vietnamese deployment constraints and local/offline operation
- introduce Cherry Core as a reliability-oriented system rather than a new pretrained model

### 2. Related Work

- Whisper-family ASR and Vietnamese fine-tuning
- Whisper hallucination studies under silence or non-speech
- runtime-efficient local inference for Whisper-family models
- optional transcript stabilization and denoising tools

### 3. System Design

Describe the current stack in repository terms:

- orchestration: `application/services/stt_web_pipeline.py`
- conservative Whisper runtime: `infrastructure/adapters/asr/whisperv2_adapter.py`
- Stable-TS option: `infrastructure/adapters/asr/stablets_adapter.py`
- post-filtering: `infrastructure/adapters/asr/hallucination_filter.py`
- VAD: `infrastructure/adapters/vad/silero_adapter.py`
- Vietnamese ASR baseline: `infrastructure/adapters/asr/phowhisper_adapter.py`

### 4. PhoWhisper.cpp Runtime Study

- describe why the study matters for CPU-only and local deployment
- distinguish conversion and quantization from training
- describe toolkit support now present in Cherry Core:
  - `scripts/phowhisper_cpp_experiment.py`
  - `research/phowhisper_cpp/benchmarking.py`
  - `research/phowhisper_cpp/evidence.py`
  - `research/phowhisper_cpp/transfer.py`

### 5. Experimental Setup

- local Vietnamese benchmark slice for system behavior
- long-form files for chunk-stitch stability
- CPU-only benchmark lane for `PhoWhisper.cpp`
- metrics: `WER`, `CER`, `RTF`, repeated n-gram hits, hypothesis/reference length ratio, qualitative failure excerpts

### 6. Results and Discussion

- show what layers help and what they do not fix
- separate verified local evidence from pilot/internal evidence
- explicitly discuss dataset constraints and external validity limits

### 7. Limitations and Threats to Validity

- small local benchmark slice
- synthetic long-form risk in older private benchmarks
- no full controlled evaluation yet for the LLM investigative layer
- varying domain mismatch between public benchmarks and target use cases

## Low-Cost Experiment Set

### Experiment A: Layered mitigation ablation

Run on 3 to 10 curated Vietnamese files:

1. PhoWhisper baseline
2. `+ VAD`
3. `+ conservative decode path`
4. `+ Stable-TS`
5. `+ BoH/delooping`

Collect:

- `WER`
- `CER`
- repeated 3-gram and 8-gram counts
- word count ratio
- notable qualitative errors

This is the highest-value low-cost experiment for the paper.

### Experiment B: Long-form stability

Use the curated long-form slice already represented by `audio_20.wav` and related benchmark files.

Primary outcomes:

- duplication at chunk boundaries
- repeated phrase loops
- omission risk through under-length outputs
- transcript readability after optional filtering

### Experiment C: PhoWhisper.cpp runtime study

Use the integrated toolkit and Colab workflow to report:

- full precision versus quantized artifacts
- CPU runtime and memory-size tradeoffs
- accuracy versus `whisper-large-v2` and `whisper-large-v3` controls

Treat the older 40-file benchmark as pilot evidence until a clean rerun is completed in the Cherry Core workflow.

## Figure and Table Plan

### Figures

- mitigation pipeline diagram
- average `WER` by model or configuration
- `WER` versus `RTF` scatter for the `PhoWhisper.cpp` lane
- long-form duplication or tail-risk visualization

### Tables

- system components and intended reliability effect
- ablation results by processing layer
- `PhoWhisper.cpp` runtime tradeoff summary
- threats to validity and claim status

## Strong Claims for the Abstract

- Cherry Core implements a layered offline Vietnamese ASR pipeline aimed at reducing hallucination and long-form instability.
- The repository contains an integrated `PhoWhisper.cpp` experiment lane for reproducible local CPU benchmarking.
- Local experiments support targeted robustness improvements, while larger claims are explicitly bounded by current dataset scale.

## Claims to Exclude from the Abstract

- any global SOTA claim
- any claim that the LLM investigative module is validated for policing outcomes
- any claim that `PhoWhisper.cpp` is a novel model or training method
