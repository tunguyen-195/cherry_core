"""
Debug Alignment Loading

Test specific loading of WhisperX alignment model to isolate errors.
"""
import logging
import traceback
import whisperx
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_alignment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    try:
        logger.info("Tentative loading alignment model for 'vi'...")
        model, metadata = whisperx.load_align_model(
            language_code="vi",
            device=device
        )
        logger.info("✅ Success loading 'vi' alignment model")
    except Exception:
        logger.error("❌ Failed loading 'vi' alignment model:")
        traceback.print_exc()

if __name__ == "__main__":
    test_load_alignment()
