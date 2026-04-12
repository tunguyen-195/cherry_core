"""
WhisperX Offline Setup Script (Fix v2)

Tải và lưu trữ tất cả models cần thiết cho WhisperX offline:
1. ASR Model (faster-whisper large-v2)
2. Alignment Model (wav2vec2-large-xlsr-53-vietnamese) - Explicitly specified
3. Diarization Model (pyannote/speaker-diarization-3.1)

Usage:
    python scripts/setup_whisperx_offline.py

Requires:
    - HF_TOKEN environment variable (for pyannote models)
    - whisperx installed
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
WHISPERX_DIR = MODELS_DIR / "whisperx"


def setup_directories():
    """Create model directories."""
    WHISPERX_DIR.mkdir(parents=True, exist_ok=True)
    (WHISPERX_DIR / "asr").mkdir(exist_ok=True)
    (WHISPERX_DIR / "alignment").mkdir(exist_ok=True)
    (WHISPERX_DIR / "diarization").mkdir(exist_ok=True)
    logger.info(f"✅ Created model directories at: {WHISPERX_DIR}")


def download_asr_model(model_size: str = "large-v2"):
    """
    Download faster-whisper ASR model.
    """
    logger.info(f"📥 Downloading ASR model: faster-whisper {model_size}...")
    
    try:
        from faster_whisper import WhisperModel
        
        # This will download and cache the model
        model = WhisperModel(
            model_size,
            device="cpu", 
            compute_type="int8",
            download_root=str(WHISPERX_DIR / "asr")
        )
        
        logger.info(f"✅ ASR model downloaded: {model_size}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download ASR model: {e}")
        return False


def download_alignment_model():
    """
    Download wav2vec2 alignment model for Vietnamese.
    Explicitly using 'nguyenvulebinh/wav2vec2-base-vietnamese-250h' or similar valid model.
    WhisperX default 'vi' might fail if hub is flaky.
    """
    logger.info("📥 Downloading alignment model (wav2vec2)...")
    
    try:
        import whisperx
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try explicit model name if 'vi' fails
        # Using the model WhisperX typically maps 'vi' to:
        model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h" 
        
        # Manually load pipeline to force download to specific dir if possible,
        # but whisperx.load_align_model handles it internally.
        # We'll rely on it caching to ~/.cache/huggingface usually, 
        # but we can try to force it or just verify it loads.
        
        align_model, metadata = whisperx.load_align_model(
            language_code="vi",
            device=device,
            model_name=model_name 
        )
        
        logger.info(f"✅ Alignment model downloaded: {model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download alignment model: {e}")
        
        # Fallback to generic 'vi'
        try:
            logger.info("   Retrying with generic 'vi' code...")
            align_model, metadata = whisperx.load_align_model(
                language_code="vi",
                device=device
            )
            logger.info("✅ Alignment model downloaded (generic vi)")
            return True
        except Exception as e2:
            logger.error(f"   ❌ Retry failed: {e2}")
            return False


def download_diarization_model():
    """
    Download pyannote diarization model.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not set!")
        return False
    
    logger.info("📥 Downloading diarization model: pyannote/speaker-diarization-3.1...")
    
    try:
        import whisperx
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        
        logger.info("✅ Diarization model downloaded")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download diarization model: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("WhisperX Offline Setup (Fix v2)")
    logger.info("=" * 60)
    
    setup_directories()
    
    download_asr_model("large-v2")
    download_alignment_model()
    download_diarization_model()
    
    logger.info("\n✅ Setup attempt complete. Check logs for errors.")

if __name__ == "__main__":
    main()
