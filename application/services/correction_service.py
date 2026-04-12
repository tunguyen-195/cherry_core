"""
LLM-based ASR Error Correction Service (Standardized).

Multi-Stage Correction Pipeline:
    1. Rule-based Phonetic Correction (fast, deterministic)
    2. ProtonX Spelling Correction (Seq2Seq)
    3. LLM Contextual Correction (semantic understanding)

Refactored to Clean Architecture (Application Layer).
"""
import logging
import json
from pathlib import Path
from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter
from core.config import BASE_DIR
from application.services.phonetic_corrector import get_phonetic_corrector

logger = logging.getLogger(__name__)

# Optimized prompt for contextual ASR correction
ASR_CORRECTION_BASE_PROMPT = """Bạn là chuyên gia sửa lỗi ASR (nhận dạng giọng nói) tiếng Việt.
Nhiệm vụ: Dùng SUY LUẬN NGỮ CẢNH để sửa lỗi sai âm (homophone errors).

Dữ liệu tham khảo (Ưu tiên):
{custom_vocab}

Hướng dẫn suy luận:
1. Phân tích ngữ cảnh của câu để phát hiện từ vô lý (Ví dụ: "xương hô" trong câu chào hỏi -> phải là "xưng hô").
2. Đối chiếu âm thanh: Từ sai thường có phát âm gần giống từ đúng.
3. TUYỆT ĐỐI KHÔNG thêm thắt nội dung ý nghĩa mới.
4. KHÔNG thay đổi cấu trúc câu nếu không cần thiết.

Ví dụ sửa lỗi (Tham khảo):
- "tiện xương hô" -> "tiện xưng hô" (ngữ cảnh hỏi tên)
- "xá phòng" -> "giá phòng" (ngữ cảnh nói về tiền)
- "G.W.Mirrors" -> "GW Marriott" (tên riêng từ điển)

Đầu vào:
{transcript}

Văn bản đã sửa (chỉ trả về text):"""

class CorrectionService:
    """
    Generalized ASR error correction using Vietnamese LLM.
    Layer: Application Service
    """
    
    def __init__(self, model_name: str = None, device: str | None = None):
        self._llm = None
        self._model_name = model_name
        self._device = device
        self._vocab_data = {}
        self._load_custom_vocab()
        
    def _load_custom_vocab(self):
        """Load comprehensive custom vocabulary (huge JSON)."""
        try:
            # Resolving path using core.config
            vocab_path = BASE_DIR / "assets" / "vocab" / "vn_general_vocabulary.json"
            
            if vocab_path.exists():
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self._vocab_data = json.load(f)
                logger.info(f"Loaded Comprehensive Vocabulary ({self._vocab_data.get('metadata', {}).get('total_count', 0)} terms).")
            else:
                logger.warning(f"Comprehensive vocabulary file not found at {vocab_path}")
        except Exception as e:
            logger.error(f"Failed to load custom vocabulary: {e} - Using empty dict.")
            self._vocab_data = {}

    def _get_relevant_vocab(self, transcript: str) -> str:
        """
        Dynamically select relevant terms based on transcript content (Keyword Matching).
        """
        if not self._vocab_data or "categories" not in self._vocab_data:
            return ""

        relevant_terms = []
        transcript_lower = transcript.lower()
        categories = self._vocab_data["categories"]

        # 1. Hospitality Check
        hospitality_triggers = ["khách sạn", "đặt phòng", "check-in", "check-out", "lễ tân", "resort"]
        if any(t in transcript_lower for t in hospitality_triggers):
            if "hospitality" in categories:
                relevant_terms.extend(categories["hospitality"][:30])
                
        # 2. Security/Crime/Drugs Check
        security_triggers = ["công an", "cảnh sát", "ma túy", "tội phạm", "vụ án", "bắt giữ", "kết án", "truy nã", "hình sự"]
        if any(t in transcript_lower for t in security_triggers):
            if "security_police" in categories:
                relevant_terms.extend(categories["security_police"])
            if "drugs" in categories:
                relevant_terms.extend(categories["drugs"])
            if "slang_underworld" in categories: 
                relevant_terms.extend(categories["slang_underworld"])

        # 3. Ethnic/Religion Check
        ethnic_triggers = ["dân tộc", "đồng bào", "nhà rông", "cồng chiêng", "tôn giáo", "đức tin"]
        if any(t in transcript_lower for t in ethnic_triggers):
             if "ethnic_religion" in categories:
                relevant_terms.extend(categories["ethnic_religion"])

        # 4. Slang/GenZ Check
        slang_triggers = ["u là trời", "khum", "xin lũi", "gét gô", "xu cà na", "chằm zn"] 
        if any(t in transcript_lower for t in slang_triggers):
             if "slang_genz" in categories:
                relevant_terms.extend(categories["slang_genz"][:20])

        if not relevant_terms:
            return ""
            
        unique_terms = list(set(relevant_terms))
        return ", ".join(unique_terms[:100])

    def _ensure_llm(self):
        """Lazy load LLM adapter."""
        if not self._llm:
            logger.info("Initializing LLM Correction Service (Generalized)...")
            # Using LlamaCppAdapter
            self._llm = LlamaCppAdapter(model_type="vistral", device=self._device)
            if not self._llm.load():
                raise RuntimeError("Failed to load LLM for ASR correction.")
            logger.info("✅ LLM Correction Service ready.")
    
    def correct(self, transcript: str) -> str:
        """
        Apply multi-stage correction to ASR transcript.
        
        Pipeline:
            1. Rule-based Phonetic Correction (fast, deterministic)
            2. LLM Contextual Correction (semantic understanding)
        """
        if not transcript or len(transcript.strip()) < 10:
            return transcript
        
        # Stage 1: Rule-based Phonetic Correction (FAST)
        phonetic_corrector = get_phonetic_corrector()
        text_after_phonetic = phonetic_corrector.correct(transcript)
        
        # Stage 2: LLM Contextual Correction
        self._ensure_llm()
        
        try:
            vocab_text = self._get_relevant_vocab(text_after_phonetic)
            if not vocab_text:
                vocab_text = "Không có"
                
            prompt = ASR_CORRECTION_BASE_PROMPT.format(
                custom_vocab=vocab_text,
                transcript=text_after_phonetic
            )
            
            # Use LLM to correct (temp=0.3)
            corrected = self._llm.generate(
                prompt,
                max_tokens=len(transcript) + 200,
                temperature=0.3
            )
            
            # Validation
            if corrected:
                ratio = len(corrected) / len(transcript)
                if 0.7 <= ratio <= 1.5:
                    logger.info(f"LLM Correction applied: {len(transcript)} -> {len(corrected)} chars")
                    return corrected.strip()
                elif ratio > 1.5:
                    logger.warning(f"LLM output expanded too much ({ratio:.2f}x).")
                    return transcript
                else:
                    logger.warning(f"LLM output too short ({ratio:.2f}x).")
                    return transcript
            else:
                logger.warning("LLM correction returned empty.")
                return transcript
                
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return transcript
