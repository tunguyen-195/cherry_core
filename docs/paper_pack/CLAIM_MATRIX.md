# Claim Matrix

Date: 2026-04-11

Legend:

- `Verified`: directly supported by repository code and/or local reproducible artifacts
- `Pilot/Internal`: supported by internal artifacts but still carrying scale or provenance caveats
- `Future Work`: not yet supported strongly enough for the main paper

| Claim | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Cherry Core implements a layered offline Vietnamese ASR pipeline with multiple mitigation stages. | Verified | `application/services/stt_web_pipeline.py`, `infrastructure/adapters/asr/*`, `infrastructure/adapters/vad/silero_adapter.py` | Safe main-method claim. |
| The `whisper-v2` path no longer depends on `openai-whisper` and avoids the NumPy 2.4 incompatibility that affected the older stack. | Verified | `infrastructure/adapters/asr/whisperv2_adapter.py` | Safe engineering claim. |
| The `whisper-v2` adapter uses conservative decoding parameters intended to reduce hallucination and repetition. | Verified | `infrastructure/adapters/asr/whisperv2_adapter.py` | Phrase as design intent plus observed behavior, not universal proof. |
| Stable-TS is available as an optional fully offline refinement stage. | Verified | `infrastructure/adapters/asr/stablets_adapter.py` | Safe functionality claim. |
| The Bag-of-Hallucinations plus delooping filter is implemented as an optional post-processing layer. | Verified | `infrastructure/adapters/asr/hallucination_filter.py` | Cite repository implementation and literature separately. |
| Local system tests show zero repeated 8-gram hits on the selected long-form case after the current chunking path. | Verified | `output/benchmarks/selected_system_tests_2026-04-10_final.json` | Narrow claim only. Do not generalize beyond the tested slice. |
| Local 3-file PhoWhisper benchmark results on a long-form slice average about 0.1969 WER and 0.139 RTF. | Verified | `output/benchmark_vi_longform_sample3.json` | Use only with file-count disclosure. |
| `PhoWhisper.cpp` is a valid systems/runtime contribution for the paper. | Verified | `scripts/phowhisper_cpp_experiment.py`, `research/phowhisper_cpp/*`, `docs/PHOWHISPER_CPP_*` | Positioning claim, not a benchmark claim. |
| A private 40-file q5 benchmark suggests `pho_large_q5` can outperform `whisper_large_v2_q5` and `whisper_large_v3_q5` on the studied synthetic long-form slice. | Pilot/Internal | `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/artifacts/benchmarks/whispercpp/q5_40_long/run_canonical/benchmark_report.json` | Must disclose risk flags and treat as pilot evidence. |
| Cherry Core is state of the art for Vietnamese ASR. | Future Work | None | Do not claim. |
| The LLM investigative module is scientifically validated for forensic decision-making. | Future Work | None | Keep as application/demo only. |
| GPU-first with CPU fallback is the intended deployment policy for the product. | Future Work | Product requirement from project direction | Keep as roadmap or implementation policy, not an evaluated paper result. |

## Allowed Headline Claims

- Cherry Core combines multiple offline mitigation layers for Vietnamese ASR reliability.
- Cherry Core includes an integrated `PhoWhisper.cpp` experiment lane for local CPU deployment study.
- Local benchmark artifacts support targeted robustness claims on a limited Vietnamese test slice.

## Disallowed Headline Claims

- Cherry Core is the best Vietnamese ASR system overall.
- `PhoWhisper.cpp` is a new model.
- The LLM intelligence output is validated investigative evidence.
- A single mitigation setting solves hallucination across all domains.
