# Manuscript Draft

Date: 2026-04-11

## Title

Towards Reliable Offline Vietnamese ASR: A Layered Mitigation Pipeline with a PhoWhisper.cpp Runtime Study

## Abstract

Offline automatic speech recognition for Vietnamese remains difficult in real deployments because practical systems must balance transcription accuracy, hallucination control, long-form stability, and local execution constraints. In this work, we present Cherry Core, an offline Vietnamese ASR system built around a layered reliability-oriented pipeline rather than a newly pretrained base model. The system combines Vietnamese Whisper-family transcription, optional voice activity detection, conservative decoding, repetition cleanup, optional Stable-TS refinement, and optional Bag-of-Hallucinations post-filtering. In parallel, we introduce a reproducible `PhoWhisper.cpp` experiment lane that studies how a Vietnamese fine-tuned Whisper checkpoint can be converted, quantized, and benchmarked inside a local C/C++ runtime for CPU-oriented deployment.

Our current repository evidence supports two main findings. First, Cherry Core already implements a multi-stage mitigation stack that reduces several known practical failure modes, including silence-triggered false output, repeated phrase loops, and unstable long-form stitching. On a targeted local system-validation set, the current long-form case produced zero repeated 8-gram hits after chunk merging, while the selected clean short-form case achieved exact normalized transcription. Second, the integrated `PhoWhisper.cpp` research lane provides a defensible systems contribution for local inference research. In a preliminary 40-file pilot benchmark from an earlier research workspace, a quantized `PhoWhisper` runtime outperformed `whisper-large-v2` and `whisper-large-v3` controls on a synthetic Vietnamese long-form slice, although those results remain pilot-only due to dataset caveats.

We argue that this framing is scientifically stronger than claiming a new Vietnamese ASR model. The main contribution is a reliability-oriented system design, grounded in current hallucination literature and coupled with a reproducible local runtime study. We conclude with a low-cost evaluation roadmap that can elevate current pilot observations into a more journal-ready Vietnamese ASR benchmark package.

## 1. Introduction

Speech-to-text systems often fail in ways that are more damaging than ordinary substitution errors. Sequence-to-sequence models may emit repeated loops, fabricate speech in silent regions, or carry unstable context across long recordings. These problems are particularly relevant in offline and field-deployable settings, where external cloud services are unavailable or undesirable and robustness matters more than leaderboard-style optimization.

Vietnamese ASR introduces an additional layer of difficulty. Domain mismatch, named entities, diacritics, colloquial speech, and limited high-quality public evaluation conditions all complicate model selection. While PhoWhisper established a strong Vietnamese fine-tuned baseline, deployment teams still face practical questions that the base model alone does not answer: how to suppress silence-induced false text, how to stabilize long-form transcripts, how to avoid repetition loops, and how to run the system locally on hardware that may vary between GPU-equipped workstations and CPU-only environments.

Cherry Core addresses this problem as a systems design problem. The repository does not claim a new acoustic architecture. Instead, it assembles a layered pipeline that targets known ASR failure modes while remaining offline-first. The design includes a Vietnamese fine-tuned model path, an alternate conservative `faster-whisper` path, VAD-driven preprocessing, optional transcript stabilization, and explicit post-processing for repeated or suspicious output.

This paper also treats local runtime portability as a scientific and engineering topic in its own right. A fine-tuned Vietnamese Whisper checkpoint may perform well in Python or Hugging Face runtime, but practical local deployment often benefits from a compiled C/C++ inference path. For this reason, we position `PhoWhisper.cpp` not as a new model, but as a reproducible runtime study involving conversion, quantization, and CPU-focused benchmarking.

The paper makes two bounded contributions:

1. A layered offline Vietnamese ASR pipeline focused on reducing hallucination-prone and long-form-unstable behavior.
2. A reproducible `PhoWhisper.cpp` experiment lane for evaluating local CPU-oriented deployment tradeoffs.

The paper does not claim state of the art across all Vietnamese ASR benchmarks, and it does not treat downstream LLM-based intelligence summarization as a validated scientific contribution. Those boundaries are important for keeping the work technically honest and publication-ready.

## 2. Related Work

PhoWhisper demonstrated that Whisper can be adapted effectively to Vietnamese through targeted fine-tuning and reported strong results on Vietnamese benchmarks such as VIVOS and VLSP [R1, R2]. This establishes a credible Vietnamese ASR baseline for offline work, but benchmark performance alone does not resolve practical deployment reliability.

Recent work has documented hallucination behavior in Whisper-family systems under silence, non-speech, and related distribution shift conditions. Barański et al. investigated hallucinations induced by non-speech audio and highlighted the concentration of recurring false outputs and looping behavior [R5]. Calm-Whisper further analyzed hallucination-related behaviors and mitigation ideas within the Whisper family [R6]. These studies motivate the use of explicit guardrails such as silence filtering, decoding constraints, and repetition cleanup.

On the runtime side, `faster-whisper` provides a practical CTranslate2-based implementation of Whisper-compatible inference and explicitly supports Transformers-compatible fine-tuned models [R3]. This makes it suitable for stable offline deployment and a useful replacement for stacks that depend on older `openai-whisper` runtime assumptions. Meanwhile, `whisper.cpp` provides a C/C++ inference path that has become important for local, CPU-friendly deployment [R4]. However, moving a fine-tuned Hugging Face checkpoint into this runtime is a deployment and tooling problem, not a training contribution by itself.

Additional tools such as Stable-TS contribute transcript and timestamp stabilization [R7], while Silero VAD offers strong local speech detection for filtering silence before transcription [R8]. Cherry Core draws from these strands to build a reliability-oriented Vietnamese ASR stack rather than proposing yet another standalone base model.

## 3. System Overview

Cherry Core is organized as a file-oriented offline transcription pipeline. The orchestration layer in `application/services/stt_web_pipeline.py` normalizes incoming audio, optionally applies VAD, runs ASR, applies optional filtering or correction stages, and writes intermediate artifacts for inspection. This design makes the pipeline suitable for analysis, benchmarking, and future reproducibility packaging.

### 3.1 Vietnamese baseline path

The primary Vietnamese model path uses `PhoWhisperAdapter` in `infrastructure/adapters/asr/phowhisper_adapter.py`. This adapter loads a local PhoWhisper checkpoint, prioritizes SafeTensors storage, and handles chunked processing for recordings longer than 30 seconds. The choice of PhoWhisper is motivated by its established Vietnamese benchmark strength rather than by any new training contribution in Cherry Core.

### 3.2 Conservative Whisper-family path

Cherry Core also includes a `whisper-v2` path backed by `faster-whisper` rather than the older `openai-whisper` runtime. The adapter in `infrastructure/adapters/asr/whisperv2_adapter.py` uses several conservative decoding choices intended to reduce unstable output:

- `beam_size = 5`
- `best_of = 5`
- `temperature = 0.0`
- `condition_on_previous_text = False`
- `compression_ratio_threshold = 2.0`
- `log_prob_threshold = -1.0`
- `no_speech_threshold = 0.5`
- optional VAD filtering
- `hallucination_silence_threshold = 1.5` when supported by the installed backend

This path also applies repetition cleanup over the returned segment text. The contribution here is not novelty in individual parameters, but their systematic use as a reliability-oriented default profile.

### 3.3 Voice activity detection

The Silero VAD adapter in `infrastructure/adapters/vad/silero_adapter.py` implements a conservative speech-preservation policy. The threshold and duration settings are chosen to reduce the chance of discarding short or low-energy speech. Silence removal before ASR is motivated directly by hallucination literature and by the practical need to reduce false output on non-speech regions.

### 3.4 Transcript stabilization and post-filtering

Cherry Core includes two optional cleanup layers after raw transcription. First, `StableTsAdapter` in `infrastructure/adapters/asr/stablets_adapter.py` provides an offline Stable-TS refinement path that uses the same local `faster-whisper` model family for consistency. Second, `HallucinationFilter` in `infrastructure/adapters/asr/hallucination_filter.py` implements a Bag-of-Hallucinations and delooping strategy inspired by current Whisper hallucination research. These layers target different failure modes: timestamp and segmentation instability on the one hand, and stereotyped or repeated false text on the other.

### 3.5 Downstream correction and analysis

The repository also includes domain-aware correction and LLM-based investigative summarization. These are valuable for product workflows, especially analyst-facing interpretation of transcripts, but they are not yet backed by a dedicated controlled evaluation protocol. Therefore, they should be presented as downstream application components rather than core scientific contributions in the current manuscript.

## 4. PhoWhisper.cpp as a Runtime Study

One of the most promising differentiators for the paper is the treatment of `PhoWhisper.cpp` as a reproducible systems lane. The key point is that `PhoWhisper.cpp` is not a separate model trained from scratch. It is a runtime-specific artifact derived from a fine-tuned PhoWhisper checkpoint through conversion and, optionally, quantization.

Cherry Core now includes a dedicated toolkit for this workflow:

- `scripts/phowhisper_cpp_experiment.py`
- `research/phowhisper_cpp/workspace.py`
- `research/phowhisper_cpp/benchmarking.py`
- `research/phowhisper_cpp/evidence.py`
- `research/phowhisper_cpp/transfer.py`

This lane supports workspace initialization, dataset materialization, conversion and quantization steps, benchmark execution, evidence rendering, and packaging of results for local archival or Colab-based execution. This is a meaningful contribution because local CPU deployment remains a real requirement in many offline settings, and fine-tuned Vietnamese checkpoints are not always available in ready-made compiled runtime formats.

The scientific value of this lane lies in careful evaluation of tradeoffs:

- accuracy relative to Whisper controls
- runtime cost under CPU inference
- artifact size after quantization
- stability proxies such as repeated phrase ratios or long-form degradation

That framing is more defensible than claiming architectural novelty and aligns the work with deployment-focused ASR systems research.

## 5. Current Evidence and Preliminary Results

The current repository already contains several artifacts that are suitable for carefully bounded reporting.

### 5.1 Selected system-validation cases

The file `output/benchmarks/selected_system_tests_2026-04-10_final.json` records three targeted checks:

1. A short clean Vietnamese clip (`audio_01.wav`) transcribed with normalized exact match, yielding `WER = 0.0` and `CER = 0.0`.
2. A short VAD availability case (`audio_09.wav`) where preprocessing reduced duration from `5.38s` to `3.69s`.
3. A long-form chunk-stitching case (`audio_20.wav`) with `WER = 0.1`, `CER = 0.0898`, and zero repeated 8-gram hits.

These are not broad benchmark results. Their value lies in confirming that the current implementation handles representative pipeline behaviors and that the long-form stitching path is not trivially collapsing into repeated overlap text on the tested case.

### 5.2 Local PhoWhisper long-form slice

The file `output/benchmark_vi_longform_sample3.json` captures a 3-file PhoWhisper benchmark on long-form Vietnamese audio. Across files `audio_10.wav`, `audio_30.wav`, and `audio_49.wav`, the average metrics are:

- `WER = 0.1969`
- `CER = 0.1929`
- `RTF = 0.139`

These values should be reported with the explicit note that they come from a small, repository-local slice. They are useful as a baseline snapshot for Cherry Core's current long-form behavior, not as a benchmark-wide conclusion.

### 5.3 Preliminary PhoWhisper.cpp pilot evidence

An earlier D12 research workspace contains a larger pilot benchmark artifact at `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/artifacts/benchmarks/whispercpp/q5_40_long/run_canonical/benchmark_report.json`. In that study, a quantized `pho_large_q5` runtime achieved:

- average `WER = 0.0811`
- average `CER = 0.0720`
- average `RTF = 0.4514`

on a 40-file Vietnamese long-form slice, outperforming the benchmarked `whisper_large_v2_q5` and `whisper_large_v3_q5` controls in average WER. However, the dataset report in the same artifact explicitly flags the slice as:

- `synthetic_longform_dataset`
- `partial_source_metadata`
- `narrow_duration_band`

Because of these caveats and because the benchmark currently lives outside the main Cherry Core evidence chain, this result should be treated as pilot/internal evidence only. It is strong enough to motivate a confirmatory rerun inside Cherry Core, but not strong enough to support an unqualified headline claim.

## 6. Discussion

The strongest paper narrative is therefore a bounded one. Cherry Core is not best presented as a novel Vietnamese pretrained ASR model. Instead, it is a practical reliability-oriented system that combines known mitigation layers in a coherent offline architecture and studies how Vietnamese fine-tuned Whisper can be deployed in a compiled runtime.

This framing has several advantages.

First, it aligns with the code that actually exists. The repository contains multiple implemented reliability stages, an explicit experiment toolkit, and preserved benchmark artifacts. The system contribution is real and inspectable.

Second, it avoids overclaiming. Many project documents in long-lived research repositories mix proposal language with validated results. By separating verified evidence from pilot evidence, Cherry Core can present an honest and technically credible story.

Third, it supports a stronger submission strategy. A paper that claims to solve Vietnamese ASR globally will be easy to challenge. A paper that claims to present a layered offline reliability pipeline plus a reproducible local runtime study is narrower but much more defensible.

## 7. Recommended Low-Cost Evaluation to Strengthen the Paper

The next experiments should be small enough to run without major resource disruption, yet structured enough to raise confidence in the manuscript.

### 7.1 Layered mitigation ablation

Use 3 to 10 curated Vietnamese files and compare:

1. raw PhoWhisper
2. `+ VAD`
3. `+ conservative decode path`
4. `+ Stable-TS`
5. `+ BoH/delooping`

Measure:

- `WER`
- `CER`
- repeated n-gram hits
- hypothesis/reference word-count ratio
- qualitative error types

This experiment most directly supports the paper's main contribution.

### 7.2 Confirmatory PhoWhisper.cpp rerun

Reuse the integrated toolkit and Colab workflow already documented in the repository. The goal is not a giant benchmark, but a clean confirmatory rerun with:

- explicit dataset provenance
- archived conversion manifest
- quantization settings
- generated figures and raw metrics stored under Cherry Core outputs

Even a smaller rerun can be enough to move the `PhoWhisper.cpp` lane from promising pilot evidence to a directly reproducible result.

### 7.3 Long-form robustness review

Add a small long-form error review focused on:

- chunk-boundary duplication
- phrase loops
- under-length outputs that may indicate omissions
- false speech in silent spans

This produces the kind of qualitative evidence that often strengthens ASR systems papers beyond aggregate WER alone.

## 8. Limitations

The current evidence base remains limited in several ways.

First, the strongest local metrics come from a narrow benchmark slice and a small number of manually selected system cases. Second, the larger `PhoWhisper.cpp` evidence currently depends on an older external workspace and a synthetic long-form dataset with explicit caveats. Third, the downstream LLM-based investigative layer lacks a controlled annotation protocol and therefore should not yet be framed as scientifically validated. Finally, no claim in the current manuscript should be interpreted as state-of-the-art dominance across all Vietnamese ASR conditions.

These limitations do not invalidate the work. Instead, they define the boundary within which the current contribution remains credible.

## 9. Conclusion

Cherry Core provides a realistic and technically defensible basis for a paper on reliable offline Vietnamese ASR. Its main strength is not the invention of a new base model, but the integration of multiple mitigation layers into an inspectable offline pipeline and the addition of a reproducible `PhoWhisper.cpp` runtime study. Current repository evidence already supports several bounded claims about system behavior, and a modest follow-up benchmark campaign could substantially strengthen the submission.

The most publication-ready position is therefore clear: present Cherry Core as a reliability-oriented Vietnamese ASR system with a local runtime deployment study, keep the claim set narrow, and treat broader ambitions such as state-of-the-art assertion or LLM investigative validation as future work.

## References

- [R1] PhoWhisper paper, https://arxiv.org/abs/2406.02555
- [R2] PhoWhisper repository, https://github.com/VinAIResearch/PhoWhisper
- [R3] faster-whisper repository, https://github.com/SYSTRAN/faster-whisper
- [R4] whisper.cpp repository, https://github.com/ggml-org/whisper.cpp
- [R5] Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio, https://arxiv.org/abs/2501.11378
- [R6] Calm-Whisper, https://arxiv.org/abs/2505.12969
- [R7] stable-ts repository, https://github.com/jianfch/stable-ts
- [R8] Silero VAD repository, https://github.com/snakers4/silero-vad
