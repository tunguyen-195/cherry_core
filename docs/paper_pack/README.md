# Cherry Core Paper Pack

Date: 2026-04-11

## Purpose

This paper pack consolidates the scientifically defensible story for Cherry Core.

The paper should be framed around two co-main contributions:

1. An offline Vietnamese ASR pipeline with layered hallucination-mitigation and transcript-stability controls.
2. A reproducible `PhoWhisper.cpp` deployment and benchmarking lane for local CPU inference.

The LLM-based investigative summary is useful for the product, but it should stay in the paper as a downstream application only, not as a core scientific contribution.

## Canonical Documents

- `PAPER_STORYLINE_AND_EXPERIMENTS.md`
  - paper framing, title direction, experiment structure, and low-cost study plan
- `MANUSCRIPT_DRAFT.md`
  - first full manuscript-ready draft grounded in current evidence
- `EXPERIMENT_EXECUTION_PLAYBOOK.md`
  - concrete low-cost benchmark plan to strengthen the paper
- `CLAIM_MATRIX.md`
  - allowed claims, disallowed claims, and evidence level
- `EVIDENCE_REGISTRY.md`
  - every important metric, figure source, and artifact path
- `SOURCE_AUDIT.md`
  - which existing docs to keep, supersede, or use as reference only
- `REFERENCES.md`
  - paper-safe reference list and citation discipline

## Core Technical Story

### Contribution A: Vietnamese ASR with layered mitigation

The strongest evidence-backed parts of Cherry Core are:

- Silero VAD preprocessing before ASR
- conservative `faster-whisper` decoding in the `whisper-v2` adapter
- repetition cleanup
- optional Stable-TS timestamp and transcript stabilization
- optional Bag-of-Hallucinations plus delooping post-filter
- optional domain-specific correction and LLM correction layers

This stack is already visible in the repository code and partially supported by local benchmark artifacts.

### Contribution B: PhoWhisper.cpp as a systems/runtime contribution

`PhoWhisper.cpp` should be positioned as:

- conversion of a Vietnamese fine-tuned Whisper checkpoint to a C/C++ runtime format
- optional quantization for CPU deployment
- benchmarking of accuracy, runtime, and hallucination proxies on Vietnamese audio

`PhoWhisper.cpp` should not be positioned as:

- a newly fine-tuned model
- a new acoustic architecture
- an official VinAI release

## Deployment Policy for the Project Narrative

Use this deployment policy consistently in docs and future implementation work:

- primary runtime: GPU-first
- fallback runtime: CPU when GPU is unavailable or fails
- special research lane: `PhoWhisper.cpp` for local CPU/offline benchmarking and deployment study

This policy helps separate the production path from the paper's systems-study path.

## Source Baseline

Official or primary sources that are safe to cite directly:

- PhoWhisper paper: https://arxiv.org/abs/2406.02555
- PhoWhisper repository: https://github.com/VinAIResearch/PhoWhisper
- faster-whisper repository: https://github.com/SYSTRAN/faster-whisper
- whisper.cpp repository: https://github.com/ggml-org/whisper.cpp
- Whisper hallucination study: https://arxiv.org/abs/2501.11378
- Calm-Whisper: https://arxiv.org/abs/2505.12969

Keep community models and older private experiments as supporting evidence only unless they are independently reproduced inside Cherry Core.

## Writing Discipline

Allowed narrative:

- "Cherry Core implements multiple layers intended to reduce false speech, repetition, and unstable long-form transcript behavior."
- "Cherry Core includes an offline PhoWhisper.cpp benchmark lane for local CPU deployment study."
- "Local evidence supports selected robustness claims on a limited Vietnamese benchmark slice."

Avoid these statements unless new evidence is produced:

- "Cherry Core achieves state of the art."
- "PhoWhisper.cpp is a new model."
- "The LLM investigative module is scientifically validated."
- "Any single mitigation universally improves all Vietnamese ASR cases."

## Immediate Author Workflow

1. Write the paper from `PAPER_STORYLINE_AND_EXPERIMENTS.md`.
2. Check every sentence against `CLAIM_MATRIX.md`.
3. Attach only figures and tables that are registered in `EVIDENCE_REGISTRY.md`.
4. Use `SOURCE_AUDIT.md` to avoid pulling mixed-quality claims from older proposal files.
