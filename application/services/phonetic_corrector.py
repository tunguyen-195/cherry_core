"""
Vietnamese Phonetic Corrector Service.
Rule-based pre-processor for ASR output correction.

Uses the generalized phonetic error dictionary to fix common
Vietnamese speech recognition errors BEFORE LLM correction.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from core.config import BASE_DIR

logger = logging.getLogger(__name__)


class VietnamesePhoneticCorrector:
    """
    Rule-based Vietnamese phonetic error corrector.
    
    Pipeline position: ASR → THIS → ProtonX → LLM Correction
    
    Fast, deterministic corrections for known phonetic patterns.
    Does NOT require GPU or external API calls.
    """
    
    def __init__(self, dictionary_path: Optional[Path] = None):
        """
        Args:
            dictionary_path: Path to vietnamese_phonetic_errors.json
        """
        if dictionary_path is None:
            dictionary_path = BASE_DIR / "assets" / "vocab" / "vietnamese_phonetic_errors.json"
        
        self.dictionary_path = dictionary_path
        self._corrections: List[Tuple[str, str]] = []
        self._loaded = False
    
    def _load_dictionary(self):
        """Load and compile correction patterns."""
        if self._loaded:
            return
        
        if not self.dictionary_path.exists():
            logger.warning(f"⚠️ Phonetic dictionary not found: {self.dictionary_path}")
            self._loaded = True
            return
        
        logger.info(f"📚 Loading Vietnamese phonetic dictionary...")
        
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all correction pairs from categories
        for category_name, category_data in data.get("phonetic_categories", {}).items():
            pairs = category_data.get("pairs", [])
            for pair in pairs:
                incorrect = pair.get("incorrect", "")
                correct = pair.get("correct", "")
                if incorrect and correct and incorrect != correct:
                    self._corrections.append((incorrect, correct))
        
        # Also add ASR-specific errors
        for pattern in data.get("asr_specific_errors", {}).get("patterns", []):
            incorrect = pattern.get("incorrect", "")
            correct = pattern.get("correct", "")
            if incorrect and correct and incorrect != correct:
                self._corrections.append((incorrect, correct))
        
        # English-Vietnamese phonetic
        for pattern in data.get("english_vietnamese_phonetic", {}).get("patterns", []):
            incorrect = pattern.get("incorrect", "")
            correct = pattern.get("correct", "")
            if incorrect and correct and incorrect != correct:
                self._corrections.append((incorrect, correct))
        
        # Hotel brand names
        for pattern in data.get("hotel_brand_names", {}).get("patterns", []):
            incorrect = pattern.get("incorrect", "")
            correct = pattern.get("correct", "")
            if incorrect and correct and incorrect != correct:
                self._corrections.append((incorrect, correct))
        
        # Sort by length (longer patterns first to avoid partial replacements)
        self._corrections.sort(key=lambda x: len(x[0]), reverse=True)
        
        logger.info(f"✅ Loaded {len(self._corrections)} phonetic correction patterns.")
        self._loaded = True
    
    def correct(self, text: str) -> str:
        """
        Apply rule-based phonetic corrections to text.
        
        Args:
            text: Raw ASR transcript
            
        Returns:
            Corrected text
        """
        self._load_dictionary()
        
        if not self._corrections:
            return text
        
        corrected = text
        applied_count = 0
        
        for incorrect, correct in self._corrections:
            if incorrect.lower() in corrected.lower():
                # Case-insensitive replacement preserving original case pattern
                pattern = re.compile(re.escape(incorrect), re.IGNORECASE)
                new_text = pattern.sub(correct, corrected)
                if new_text != corrected:
                    applied_count += 1
                    corrected = new_text
        
        if applied_count > 0:
            logger.info(f"🔧 Applied {applied_count} phonetic corrections.")
        
        return corrected
    
    def get_correction_count(self) -> int:
        """Return number of loaded correction patterns."""
        self._load_dictionary()
        return len(self._corrections)
    
    def add_custom_correction(self, incorrect: str, correct: str):
        """Add a custom correction pattern at runtime."""
        self._load_dictionary()
        self._corrections.insert(0, (incorrect, correct))
        logger.info(f"➕ Added custom correction: '{incorrect}' → '{correct}'")


# Singleton instance for convenience
_corrector_instance: Optional[VietnamesePhoneticCorrector] = None

def get_phonetic_corrector() -> VietnamesePhoneticCorrector:
    """Get singleton instance of VietnamesePhoneticCorrector."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = VietnamesePhoneticCorrector()
    return _corrector_instance
