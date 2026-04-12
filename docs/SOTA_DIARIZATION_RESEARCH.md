# SOTA Speaker Diarization Research (2025)

**Date**: 2026-01-11
**Subject**: Architectural Deep Dive & Breakthrough Techniques for Accuracy

## 1. Landscape Overview (2025)

The "State of the Art" (SOTA) in speaker diarization has shifted from simple clustering (K-Means/Spectral) to **End-to-End Neural Networks** and **Hybrid Systems** that handle overlapping speech and precise boundary detection.

### Leading Architectures

| System | Architecture | Key Feature | Offline Viability |
| :--- | :--- | :--- | :--- |
| **Pyannote 3.1** | PyanNet (Segmentation) + WeSpeaker (Embedding) | **Powerset Encoding** (Handles overlaps natively) | ⭐⭐⭐⭐⭐ (Excellent) |
| **NVIDIA NeMo** | TitaNet (Embedding) + MSDD (Multi-Scale Decoder) | **Multi-Scale Clustering** (Captures short & long turns) | ⭐⭐⭐⭐ (Requires GPU) |
| **EEND-EDA** | Transformer/Conformer Encoder-Decoder | **Overlap-Aware** (No clustering step needed) | ⭐⭐⭐ (Hard to train) |
| **SpeechBrain** | ECAPA-TDNN + Spectral + VBx | **Modular** (Highly customizable) | ⭐⭐⭐⭐⭐ (Current Choice) |

## 2. Secrets of "Famous Services" (AssemblyAI, Google)

Research into whitepapers from AssemblyAI and Google reveals a common "Golden Pipeline":

1. **VAD & Segmentation**: They don't just "split by silence". They use neural segmentation models to find "speaker change points" even without silence.
2. **Embedding Fusion**: They often combine multiple embeddings (e.g., TitaNet + ECAPA) or use large-scale pre-trained models (WavLM/HuBERT) as feature extractors.
3. **Global Optimization (The "Secret Sauce")**:
    * **VBx Resegmentation**: After initial clustering, they run a **Variational Bayes** pass. This effectively "re-aligns" every frame based on the global probability of the speaker model. It smoothes out "jittery" labels.
    * **Overlap Assignment**: They explicitly detect overlapping regions and assign multiple labels.

## 3. Breakthrough Techniques for Accuracy Improvement

To move Cherry Core V2 from "Good" to "SOTA Forensic Grade", we must adopt these specific techniques:

### A. Variational Bayes (VBx) Resegmentation

* **Concept**: Treat the initial clustering (Spectral) as a "draft". Use a Hidden Markov Model (HMM) where each state is a Speaker. Calculate the probability of the audio frames belonging to each speaker and find the optimal path (Viterbi/Forward-Backward).
* **Code Reference**: `BUT Speech@FIT` (Original VBx implementation) or `SpeechBrain` recipes.
* **Expected Gain**: 2-5% absolute DER reduction (Fixes "short blips" and incorrect boundary shifts).

### B. End-to-End Neural Diarization (EEND)

* **Concept**: A single neural network takes audio as input and outputs `(Time, Speaker)` directly. No separate embedding/clustering steps.
* **Challenge**: Requires fixed number of speakers or complex "Attractors" (EDA) for unknown counts. Harder to run offline on generic hardware compared to modular approaches.

### C. Overlap Detection with Powerset (Pyannote style)

* **Concept**: instead of predicting "Speaker A" or "Speaker B", predict the *set* of active speakers `{A, B}`.
* **Implementation**: Requires a specific segmentation model trained with powerset labels.

## 4. Implementation Strategy for Cherry Core V2

Given our constraint of **Strict Offline Operation** and **Modularity**, the best path is finding the "Golden Mean":

1. **Keep ECAPA-TDNN** (Robust, fast).
2. **Keep Spectral Clustering** (Auto-detects K).
3. **ADD VBx Resegmentation Layer**: This is the missing link to "smooth" the output and match famous services' precision.
4. **Refine Word-Alignment**: Ensure timestamps align with phonemes, not just VAD chunks.

**Conclusion**: We will implement **Module 8.6: VBx HMM Resegmentation** using `SpeechBrain` logic but adapted for our custom pipeline.
