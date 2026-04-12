# Reference Pack

Date: 2026-04-11

This file collects the primary references that are safe to use in the manuscript draft.

## Core Vietnamese ASR

1. PhoWhisper: Automatic Speech Recognition for Vietnamese
   - arXiv: https://arxiv.org/abs/2406.02555
   - repository: https://github.com/VinAIResearch/PhoWhisper
   - use: Vietnamese fine-tuned baseline and official benchmark context

## Runtime and Deployment

2. faster-whisper
   - repository: https://github.com/SYSTRAN/faster-whisper
   - use: CTranslate2-based Whisper inference and fine-tuned model runtime context

3. whisper.cpp
   - repository: https://github.com/ggml-org/whisper.cpp
   - use: C/C++ runtime and local CPU deployment context

## Hallucination and Reliability

4. Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio
   - arXiv: https://arxiv.org/abs/2501.11378
   - use: motivation for VAD, BoH-style filtering, and non-speech robustness discussion

5. Calm-Whisper
   - arXiv: https://arxiv.org/abs/2505.12969
   - use: supporting hallucination-mitigation literature for Whisper-family systems

## Supporting Tools

6. stable-ts
   - repository: https://github.com/jianfch/stable-ts
   - use: transcript and timestamp stabilization context

7. Silero VAD
   - repository: https://github.com/snakers4/silero-vad
   - use: offline VAD preprocessing context

## Citation Discipline

- Prefer citing the paper when a formal paper exists and the claim is scientific.
- Cite the repository when the point is about implementation, runtime support, or available tooling.
- Do not cite community conversions or mirrors as authoritative evidence for the main paper claims.
