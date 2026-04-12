# PhoWhisper C++ / whisper.cpp Research Note

Date: 2026-04-11

## Question

Can PhoWhisper be "fine-tuned into" a `PhoWhisper.cpp` model, and should ideas or code from `E:\Freelance\Research\D12_02.2026_NCKH2026\ASR` be imported into Cherry Core?

## Short Answer

No new fine-tuning step is required to create a `PhoWhisper.cpp` runtime artifact.

`PhoWhisper.cpp` should be understood as:

1. `PhoWhisper` fine-tuned in the standard Hugging Face / Transformers format.
2. Converted into a runtime-specific format:
   - `ggml` for `whisper.cpp`
   - `CTranslate2` for `faster-whisper`
3. Optionally quantized for CPU-friendly deployment.

So the technical problem is primarily **conversion + runtime benchmarking**, not training.

## External Findings

### Official PhoWhisper status

- The official PhoWhisper repository documents usage through `transformers`.
- The repo exposes benchmark results and model cards, but it does not document an official `whisper.cpp` or C++ runtime package.

Implication:

- There is no strong evidence of a first-party `PhoWhisper.cpp` distribution from VinAI.

### Official whisper.cpp status

- `whisper.cpp` is an inference runtime, not a training stack.
- The public README still documents `ggml` model conversion and quantization for standard Whisper models.
- An open `whisper.cpp` issue requests an official utility for converting Hugging Face Whisper checkpoints, especially fine-tuned `safetensors` models, into a runtime format.

Implication:

- Running a fine-tuned Whisper-family model inside `whisper.cpp` is feasible.
- But it is still not a completely first-class, one-command workflow for arbitrary Hugging Face fine-tuned checkpoints.

### Official faster-whisper status

- `faster-whisper` officially states it can convert **any Whisper models compatible with the Transformers library**, including user fine-tuned models.

Implication:

- If the goal is a stable production runtime for PhoWhisper on CPU/offline, CTranslate2 is the lower-risk path.

### Community evidence

- Community CTranslate2 packages exist for PhoWhisper, including:
  - `kiendt/PhoWhisper-large-ct2`
  - `quocphu/PhoWhisper-ct2-FasterWhisper`
- Community `ggml` conversion also exists at least for `PhoWhisper-small`.

Implication:

- The conversion path is real and has been attempted successfully by multiple independent users.
- But community artifacts should be treated as convenience references, not as authoritative production baselines.

## Review of the Old Research Project

Path reviewed:

- `E:\Freelance\Research\D12_02.2026_NCKH2026\ASR`

### What the old project was actually doing

The old project was not attempting to retrain PhoWhisper into a different architecture.

It was building a benchmark workflow around:

1. Downloading `vinai/PhoWhisper-large`
2. Converting it to `ggml` for `whisper.cpp`
3. Quantizing to `q4_0` / `q5_0`
4. Running CPU benchmarks against:
   - `PhoWhisper-large`
   - `Whisper large-v2`
   - `Whisper large-v3`

Key evidence:

- `ASR/notebooks/colab/2026-03-18_phowhisper-large-whispercpp/01_convert_quantize_phowhisper_large_whispercpp_colab.ipynb`
- `ASR/scripts/benchmark/whisper_cpp/run_whisper_cpp_q5_40_long_colab.py`
- `ASR/reports/model-selection/bao_cao_so_sanh_3_mo_hinh_q5.md`

### Strong points of the old project

1. Good research workspace hygiene
   - Clean separation of `artifacts`, `reports`, `docs`, `external`, `configs`.
   - Better traceability than an ad hoc notebook-only setup.

2. Good reproducibility mindset
   - `run_manifest.json` records dataset fingerprint, artifact hashes, model paths, and benchmark config.
   - This is worth carrying over to Cherry Core benchmark tooling.

3. Useful path registry
   - `configs/project_paths.json`
   - `src/common/paths.py`
   - These are practical and reusable.

4. Practical wrapper around a large legacy scorer
   - `scripts/benchmark/whisper_cpp/score_whisper_cpp_benchmark.py`
   - The wrapper adds dataset/experiment resolution and writes a manifest, which is the right direction.

5. Clear benchmark framing
   - The report compares **accuracy**, **hallucination proxy**, and **speed**.
   - That framing matches Cherry Core’s actual goals much better than raw WER alone.

### Weak points / risks in the old project

1. Benchmark data quality is limited
   - The canonical run is flagged as:
     - `synthetic_longform_dataset`
     - `partial_source_metadata`
     - `narrow_duration_band`
   - This makes the result useful, but not sufficient as a universal model-selection conclusion.

2. Too much Colab coupling
   - Several scripts assume `apt-get`, interactive uploads, Google Colab filesystem layout, and notebook execution.
   - That is research-convenient, not production-convenient.

3. Legacy scorer is too large to import wholesale
   - `_score_whisper_cpp_benchmark_legacy.py` is roughly 942 lines.
   - This is a warning sign: clone behavior, not just files.

4. whisper.cpp conversion path is fragile
   - The notebook includes a `safetensors` fallback and manual workaround logic.
   - That is useful evidence, but it also proves the path is brittle.

5. The old project is benchmark-first, not webapp-first
   - It does not directly match Cherry Core’s runtime/web UX architecture.

## Results Already Produced by the Old Project

The canonical 40-file CPU `q5_0` benchmark in the old project reports:

- `pho_large_q5`: Avg WER `0.0811`, Avg RTF `0.4519`
- `whisper_large_v3_q5`: Avg WER `0.2207`, Avg RTF `0.4254`
- `whisper_large_v2_q5`: Avg WER `0.2228`, Avg RTF `0.4191`

Interpretation:

- On that benchmark, quantized PhoWhisper inside `whisper.cpp` was dramatically more accurate for Vietnamese than the original large-v2/large-v3 baselines.
- Speed loss was relatively small compared to the quality gain.

Important caveat:

- This is a strong directional signal, not a universal conclusion, because the dataset is synthetic long-form and metadata is incomplete.

## Recommendation for Cherry Core

### Recommended runtime path

Primary recommendation:

1. Keep `PhoWhisper` as the Vietnamese accuracy-first baseline.
2. Prefer **CTranslate2 / faster-whisper conversion** before investing heavily in `whisper.cpp`.

Reason:

- `faster-whisper` officially supports conversion of Transformers-compatible Whisper models.
- This path is easier to automate and integrate into the current Python-based Cherry Core runtime.
- It fits the current architecture better than a direct `whisper.cpp` subprocess path.

Secondary recommendation:

3. Keep a `whisper.cpp` benchmark lane for CPU-only comparison and packaging experiments.

Reason:

- The old project already shows the benchmark direction is promising.
- But it should remain a benchmark/runtime option, not the first production integration target.

### What should be cloned or adapted

Recommended to adapt:

- `E:\Freelance\Research\D12_02.2026_NCKH2026\ASR\src\common\paths.py`
- `E:\Freelance\Research\D12_02.2026_NCKH2026\ASR\configs\project_paths.json`
- `E:\Freelance\Research\D12_02.2026_NCKH2026\ASR\scripts\benchmark\whisper_cpp\score_whisper_cpp_benchmark.py`
- The `run_manifest.json` pattern
- The benchmark report framing: accuracy + hallucination proxy + speed

Recommended to reference but not clone wholesale:

- Colab notebooks under `ASR/notebooks/colab/...`
- `run_whisper_cpp_q5_40_long_colab.py`
- `_score_whisper_cpp_benchmark_legacy.py`

Reason:

- They contain useful logic and recovery steps.
- But they are too environment-specific or too large to import directly.

Recommended not to clone as-is:

- The entire old repo structure
- Interactive upload flows
- Colab package installers
- Research-only archive materials

## Decision

If the goal is:

- **best engineering ROI now**: bring over the benchmark manifest/wrapper ideas and build a `PhoWhisper-ct2` lane first.
- **maximum CPU portability later**: keep the old `whisper.cpp` research as a second-phase runtime track.

In other words:

- `PhoWhisper.cpp` is a **conversion/runtime packaging problem**
- not a **new fine-tuning research problem**

## External Sources

- PhoWhisper official repo: https://github.com/VinAIResearch/PhoWhisper
- whisper.cpp official repo: https://github.com/ggml-org/whisper.cpp
- whisper.cpp issue on HF fine-tuned model conversion: https://github.com/ggml-org/whisper.cpp/issues/3316
- faster-whisper official repo: https://github.com/SYSTRAN/faster-whisper
- Community PhoWhisper large CT2: https://huggingface.co/kiendt/PhoWhisper-large-ct2
- Community PhoWhisper CT2 multi-variant: https://huggingface.co/quocphu/PhoWhisper-ct2-FasterWhisper
- Community PhoWhisper small ggml: https://huggingface.co/dongxiat/ggml-PhoWhisper-small
