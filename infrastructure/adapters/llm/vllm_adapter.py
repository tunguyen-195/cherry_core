import logging
from typing import Dict, Any, List, Optional
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from core.ports.llm_port import ILLMEngine
from core.config import MODELS_DIR

logger = logging.getLogger(__name__)

class VLLMAdapter(ILLMEngine):
    """
    Adapter for vLLM Engine (High-Performance).
    Uses PagedAttention for maximum throughput.
    """
    def __init__(self, model_name: str = "vistral-7b-chat-Q4_K_M.gguf"):
        # Note: vLLM usually requires unquantized models or AWQ/GPTQ.
        # GGUF support in vLLM is experimental/limited.
        # We will assume the user might switch to an AWQ model later,
        # but for now we try to load what we have or a standard HF path.
        self.model_path = str(MODELS_DIR / "vistral" / model_name)
        self.llm = None
        self.sampling_params = None

    def load(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception:
            return False
        
    def _ensure_model(self):
        if self.llm:
            return

        if LLM is None:
            raise ImportError("vLLM library not installed.")

        logger.info(f"🚀 Initializing vLLM Engine from {self.model_path}...")
        try:
            # vLLM requires a huggingface repo ID or a folder with config.json
            # GGUF is not directly supported by standard vLLM.
            # We will use the 'model_path' but vLLM might expect a directory.
            # Cycle 4 constraint: If GGUF fails, we might need to fallback.
            
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=4096,
                stop=["</s>", "User:", "System:", "<|im_end|>"]
            )
            logger.info("✅ vLLM Engine loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load vLLM: {e}")
            raise e

    def generate(self, prompt: str, max_tokens: int = 4096, grammar_path: str = None) -> str:
        self._ensure_model()
        
        # Guided Decoding (vLLM native)
        # vLLM supports 'guided_decoding_backend' but as of v0.2.0 it uses 'outlines' library
        # We can pass regex or json_schema directly if supported.
        # For now, we run standard generation.
        
        logger.info("⚡ vLLM Generating...")
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text

    def analyze(self, transcript_text: str, scenario: str) -> Dict[str, Any]:
        # Reuse existing prompt manager logic from service, 
        # but this adapter is low-level. 
        # AnalysisService calls generate().
        pass
