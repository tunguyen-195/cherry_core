# Vocabulary Optimization Strategy for ASR Correction

**Goal**: Maximize the utility of the 73k+ word custom vocabulary (`vn_general_vocabulary.json`) while ensuring the LLM remains stable, accurate, and does not hallucinate (prevent "hiểu lầm/bị lú").

## 1. Problem Analysis

- **Context Window Limit**: We cannot inject 73,000 words into the LLM prompt.
- **Hallucination Risk**: Providing too many irrelevant terms confuses the model ("lú"), leading it to force-fit those terms where they don't belong.
- **Retrieval Precision**: Simple keyword matching (current implementation) misses synonyms and can trigger false positives (e.g., "bắt" triggers drug vocab even if it's "bắt tay").

## 2. Research-Based Solutions

### A. Hybrid Retrieval Strategy (Keyword + Semantic)

Instead of relying solely on exact keyword matches, we propose a tiered approach:

1. **Tier 1: Fast Triggering (Keyword)**: Use existing keyword matching to identify *potential* domains (e.g., "khách sạn", "ma tuý").
2. **Tier 2: Contextual Filtering (Semantic - Future Upgrade)**: Use lightweight embeddings (e.g., `sentence-transformers` or `fasttext`) to verify if the transcript *actually* relates to the domain before injecting the full list.
    - *Current Action*: Refine the "Trigger List" to be more specific (avoid generic words like "bắt", "án").

### B. "Delineated" Prompt Engineering (RAG Style)

To prevent the LLM from treating the vocabulary as a "mandatory insertion list" (which causes hallucinations):

1. **Explicit Sectioning**: Clearly separate "Context/Vocabulary" from "Instructions".
2. **"Reference Only" Instruction**: Explicitly tell the LLM: *"Use these terms ONLY if phonetic matches are ambiguous. Do NOT force these terms if the audio suggests otherwise."*
3. **Negative Constraints**: Add "Anti-Hallucination" rules: "If the transcript is already clear, do not change it."

### C. Constrained Decoding (Grammar)

Use `llama.cpp`'s grammar constraints to ensure the output structure is valid, though constraining the *content* to the vocabulary is too rigid for general speech. We will rely on prompt constraints instead.

## 3. Implementation Plan (Optimization)

### Step 1: Refine Keyword Triggers (Immediate)

- Audit current triggers in `LLMCorrectionService`.
- Remove generic triggers (e.g., "bắt", "án", "người", "lễ") that cause false positive domain injection.
- Use multi-word triggers (e.g., "bắt giữ đối tượng", "vụ án", "lễ hội") for higher precision.

### Step 2: Prompt Optimization (Immediate)

- Update `ASR_CORRECTION_BASE_PROMPT` to frame the vocabulary as a **"Reference Dictionary"** rather than a "Priority List".
- Add a **"Confidence Check"** instruction.

### Step 3: Semantic Reranking (Future Cycle)

- If keyword matching remains too noisy, integrate a small local embedding model to rerank vocabulary relevance.

## 4. Expected Outcome

- **Reduced Hallucination**: LLM won't force-fit "Ma túy" into a sentence about "Ma thuật".
- **Higher Precision**: Only relevant specialized terms are injected.
- **Stability**: General conversation remains unaffected.
