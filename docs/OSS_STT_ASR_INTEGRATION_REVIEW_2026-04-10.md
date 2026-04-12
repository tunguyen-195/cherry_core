# OSS STT/ASR Integration Review

Date: 2026-04-10
Repo: `cherry_core`

## Goal

Review open-source STT/ASR projects that target higher accuracy and lower hallucination/error rates, especially for Vietnamese, and decide what is practical to integrate into the current offline-first pipeline.

## Immediate Integration

- `faster-whisper`
  - Why: CTranslate2 backend, lower memory footprint, supports `word_timestamps`, `vad_filter`, and `hallucination_silence_threshold`.
  - Fit for this repo: direct replacement for the old `openai-whisper`-based `whisper-v2` adapter.
  - Decision: integrate now as the runtime backend for `whisper-v2`.
  - Source: https://github.com/SYSTRAN/faster-whisper

- `WhisperX`
  - Why: adds batching, wav2vec2 alignment, and uses VAD plus `condition_on_prev_text=False`, which is aligned with the repo's anti-hallucination direction.
  - Fit for this repo: optional alignment/timeline mode, not the default offline diarization path.
  - Decision: keep as optional advanced backend; do not make it the default because pyannote/HF-token friction is still real for offline users.
  - Source: https://github.com/m-bain/whisperX

- `PhoWhisper`
  - Why: Vietnamese fine-tuned Whisper family and already strong for the repo's main language.
  - Fit for this repo: already integrated as the primary ASR engine.
  - Decision: keep as default Vietnamese engine and benchmark new backends against it.
  - Source: https://github.com/VinAIResearch/PhoWhisper

- `Silero VAD`
  - Why: strong speech-only filtering and already aligned with the current pipeline.
  - Fit for this repo: keep as optional preprocessing, but not always-on.
  - Decision: keep current adapter; do not move VAD to mandatory mode.
  - Source: https://github.com/snakers4/silero-vad

## Medium-Term Candidates

- `stable-ts` / `stable-ts-whisperless`
  - Why: silence suppression, timestamp stabilization, regrouping, and support for `faster-whisper`.
  - Fit for this repo: best added as an optional post-ASR cleanup/timestamp refinement stage after `whisper-v2` or `whisperx`.
  - Decision: good next integration if the team wants a switchable "stabilize transcript/timestamps" option.
  - Source: https://github.com/jianfch/stable-ts

- `DeepFilterNet`
  - Why: open-source speech enhancement / denoising that can improve ASR before VAD and decoding on noisy recordings.
  - Fit for this repo: optional preprocessor before VAD for hotline, field audio, or noisy interviews.
  - Decision: worth prototyping behind a toggle; not needed for clean benchmark audio.
  - Source: https://github.com/Rikorose/DeepFilterNet

- `Zipformer-30M-RNNT-6000h`
  - Why: Vietnamese-focused model with promising offline footprint and a route to CPU deployment through `sherpa-onnx`.
  - Fit for this repo: candidate for a dedicated Vietnamese fallback engine that is not Whisper-based.
  - Decision: benchmark later as a new adapter, not a drop-in replacement today.
  - Sources:
    - https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h
    - https://github.com/k2-fsa/sherpa-onnx

- `NVIDIA Parakeet CTC Vietnamese`
  - Why: recent Vietnamese ASR model with punctuation/capitalization support.
  - Fit for this repo: interesting for high-quality formatted transcripts, but the NeMo stack adds deployment cost.
  - Decision: watchlist only for now.
  - Source: https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese

## Not Recommended Right Now

- `PhoASR-whisper`
  - Reason: the paper is interesting, but I could not verify a public repo/model release suitable for immediate integration as of 2026-04-10.
  - Decision: keep on watchlist until there is a stable public release.
  - Source: https://aclanthology.org/2026.findings-eacl.345/

- `pyannote` as the default diarization path
  - Reason: token/license/cache friction is still too high for the repo's offline-first requirement.
  - Decision: keep it out of the default webapp path.

## Recommended Architecture Moves

1. Keep `PhoWhisper` as the default Vietnamese ASR engine.
2. Keep `whisper-v2` as the conservative fallback engine, but run it on `faster-whisper/CTranslate2`.
3. Keep `WhisperX` optional for alignment-heavy workflows.
4. Add `stable-ts` later as an optional refinement stage rather than baking it into the default path.
5. Prototype `DeepFilterNet` only for noisy-domain workloads.
6. Benchmark `Zipformer-30M-RNNT-6000h` before considering a new Vietnamese-specific backend.
