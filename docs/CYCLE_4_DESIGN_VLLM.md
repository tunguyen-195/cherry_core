# CYCLE 4 DESIGN: HIGH-PERFORMANCE SERVING WITH vLLM
**Objective**: Integrate `vLLM` (PagedAttention) into Cherry Core V2 using Modern Design Patterns.

---

## 1. RESEARCH & MOTIVATION
### 1.1 Why vLLM?
- **PagedAttention**: Manages KV cache memory efficiently (like OS virtual memory), allowing higher batch sizes.
- **Continuous Batching**: Processes incoming requests immediately rather than waiting for valid batches.
- **Guided Decoding**: vLLM supports `outlines` and regex-based decoding natively, which is superior to `llama.cpp` GBNF (Cycle 1).

### 1.2 Architecture Constraints
- **OS**: Windows (vLLM has limited/experimental support on Windows natively, often requires WSL2 or Docker). *CRITICAL RISK*: If native Windows support is flaky, we might need a Docker container or fallback to `IPEX-LLM` (Intel) or remain on `llama.cpp` for pure Windows.
- **Hardware**: GPU (CUDA) is mandatory for vLLM efficiency.

---

## 2. DESIGN PATTERNS APPLIED
### 2.1 Strategy Pattern (The Engine Interface)
We already have `ILLMEngine` (Port).
- **Existing**: `LlamaCppAdapter` (Concrete Strategy A).
- **New**: `VLLMAdapter` (Concrete Strategy B).
- **Context**: `AnalysisService` selects the strategy based on configuration (`config.USE_VLLM`).

### 2.2 Adapter Pattern
`vLLM` provides an `LLMEngine` class (Python) and an OpenAI-compatible API Server.
- **Approach A (Python Library)**: Direct import `vllm.EngineArgs`, `vllm.LLMEngine`. Good for monolithic apps.
- **Approach B (Client-Server)**: Run vLLM as a separate process (microservice) and use `AsyncHTTPClient`. This separates concerns (Stability Pattern).
> **Decision**: Using **Approach B (Client-Server)** is more "Modern Cloud-Native". It prevents LLM crashes from taking down the Controller.

### 2.3 Factory Pattern
Update `SystemFactory` to instantiate the correct Adapter:
```python
def create_llm_engine(self) -> ILLMEngine:
    if config.LLM_BACKEND == "vllm":
        return VLLMClientAdapter(url=config.VLLM_API_URL)
    return LlamaCppAdapter()
```

### 2.4 Asynchronous Design (AsyncIO)
vLLM is inherently async. Our `AnalysisService` is currently synchronous.
- **Refactoring**: We should introduce `async def analyze_transcript(...)` to fully leverage concurrent processing if we process multiple evidences.

---

## 3. IMPLEMENTATION PLAN
1.  **Environment Check**: Verify if `vllm` can be installed on this Windows environment. (High Risk).
    - *Fallback*: If vLLM fails on Windows, we design the **Remote Proxy Pattern** where the Engine runs on WSL2/Linux, and Cherry connects via HTTP.
2.  **DTOs (Data Transfer Objects)**: Define strict Pydantic models for Input/Output to decouple from specific library dicts.
3.  **Code Structure**:
    - `infrastructure/adapters/llm/vllm_client.py`
    - `core/ports/llm_port.py` (Update for Async).

---

## 4. NEXT ACTIONS
1.  **Feasibility Probe**: Try `pip install vllm` (or check compatibility).
2.  **Mocking**: If installation is complex, implement the *Interface* first using a Mock, ensuring the Design is solid before fighting dependencies.
