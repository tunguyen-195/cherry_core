# RESEARCH: GRAMMAR CONSTRAINED DECODING (GBNF) FOR FORENSIC AI

## 1. Problem Identification (The "Echoing" & "Invalid JSON" Issue)
Small Language Models (SLMs) like Vistral-7B often struggle with:
- **Instruction Echoing**: Repeating the system prompt instead of following it.
- **Broken JSON**: Generating Markdown formatting (```json) or trailing commas that break parsers.
- **Hallucination**: Inventing keys or fields not requested.

## 2. Solution: GBNF (Grammar-Based Normalization Form)
Llama.cpp supports **GBNF**, a context-free grammar syntax that *forces* the model to output tokens that match a specific structure. This is superior to "Prompt Engineering" because it operates at the **Logit Level** (token probability manipulation).

### 2.1 Theoretical Mechanism
Instead of allowing the model to pick *any* token from vocabulary V, GBNF masks all tokens that would violate the current state of the grammar stack.
- **State**: Expecting `key` -> Mask everything except `"` + strings.
- **State**: Expecting `value` -> Mask everything except numbers/strings.

### 2.2 Benchmarks (Literature Review)
- **Reliability**: Improves JSON validity from ~70% (7B models) to **100%**.
- **Latency**: Negligible overhead (< 5ms per token mask).
- **Security**: Prevents "Prompt Injection" attacks from producing unauthorized output formats.

## 3. Implementation Strategy for Cherry V2
### 3.1 GBNF Schema Design
We must define a grammar that strictly enforces the `DeepInvestigationReport` structure:
```gbnf
root ::= "{" ws "STRATEGIC_ASSESSMENT" ":" ws assessment "," ... "}"
assessment ::= "{" ws "\"executive_briefing\"" ":" ws string "," ... "}"
...
```

### 3.2 Integration Point
Modify `LlamaCppAdapter` to accept a `grammar_path` argument.
Pass this state down to the low-level `llama_cpp` generation call.

## 4. Expected Outcome (Metric for Cycle 1)
- **0% JSON Errors**.
- **0% Markdown Pollution** (No ```json fences).
- **Administrative Paragraph Enforcement**: The grammar can restrict the `executive_briefing` value to be a single string (preventing bullet points by disallowing newline characters if needed, though broad string acceptance is safer).
