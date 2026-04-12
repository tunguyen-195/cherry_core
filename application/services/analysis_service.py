import logging
import json
from typing import Optional
from application.services.prompt_manager import PromptManager
from core.ports.llm_port import ILLMEngine
from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Forensic Summarization Service.
    Bridge between Transcripts and Strategic Intelligence.
    Standardized Application Service.
    """
    def __init__(self, engine: Optional[ILLMEngine] = None, prompt_manager: Optional[PromptManager] = None):
        self.engine = engine or LlamaCppAdapter()
        self.prompt_manager = prompt_manager or PromptManager()
        self._is_loaded = False
        
    def load_model(self):
        if not self._is_loaded:
            if self.engine.load():
                self._is_loaded = True
            else:
                raise RuntimeError("Could not load LLM Engine.")

    def analyze_transcript(self, transcript: str, scenario: str = "drug_trafficking") -> dict:
        """
        Perform forensic analysis on a transcript.
        """
        self.load_model()
        
        # 1. Render Prompt (Deep Investigation)
        template_name = "deep_investigation.j2"
        
        prompt = self.prompt_manager.render_prompt(
            template_name=template_name,
            transcript=transcript,
            scenario=scenario
        )
        
        logger.info(f"Sending Deep Analysis Prompt to LLM (Scenario: {scenario})...")
        
        # 2. Inference
        raw_response = self.engine.generate(prompt)
        
        # 3. Parse JSON (Robust)
        parsed = self._parse_json_response(raw_response)
        if parsed is not None:
            return parsed

        logger.warning("Failed to parse JSON directly. Returning raw text.")
        return {"raw_output": raw_response, "error": "JSON Parse Error"}

    def _parse_json_response(self, raw_response: str) -> Optional[dict]:
        candidates = []
        text = (raw_response or "").strip()

        if not text:
            return None

        candidates.append(text)

        if "```json" in text:
            candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
        elif "```" in text:
            candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())

        if "{" in text and "}" in text:
            candidates.append(text[text.find("{"): text.rfind("}") + 1].strip())

        # Some prompts pre-fill the opening JSON fragment in the assistant turn.
        if not text.startswith("{"):
            candidates.append('{"sva_analysis": ' + text)
            candidates.append('{"threat_level": "' + text)

        for candidate in candidates:
            if not candidate:
                continue

            cleaned = candidate.strip()
            if "{" in cleaned and "}" in cleaned and not cleaned.startswith("{"):
                cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1].strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        return None
