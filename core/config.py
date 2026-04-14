import os
from pathlib import Path

# Paths
# Adjust paths relative to this file: core/config.py -> parent(core) -> parent(cherry_core)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

PHOWHISPER_PATH = MODELS_DIR / "phowhisper-safe"  # VinAI PhoWhisper for Vietnamese
WHISPER_V3_PATH = MODELS_DIR / "whisper-large-v3"
WHISPER_V2_PATH = MODELS_DIR / "whisper-large-v2"
PROTONX_PATH = MODELS_DIR / "protonx"
SPEECHBRAIN_PATH = MODELS_DIR / "speechbrain"
SILERO_PATH = MODELS_DIR / "silero"

# LLM Configuration
# Keep llama.cpp as the default local LLM backend for the supported first-run path.
# vLLM is intentionally out of the default bootstrap because it complicates Windows
# and offline-first setup. Advanced users can re-enable it later if needed.
USE_VLLM = False
LLM_MODEL_NAME = "vistral-7b-chat-Q4_K_M.gguf"

# Audio
SAMPLE_RATE = 16000

# Settings
OFFLINE_MODE = True


class ASRConfig:
    """ASR Model Selection - All models MUST be downloaded locally first"""
    
    # Engine Selection: "phowhisper" (Vietnamese SOTA) or "whisper-v2" (General)
    ENGINE = "whisper-v2"
    
    # PhoWhisper (VinAI) - SOTA for Vietnamese
    # Download at: https://huggingface.co/vinai/PhoWhisper-large
    # WER: 4.67% (VIVOS), 8.14% (CMV-Vi)
    PHOWHISPER_MODEL = "vinai/PhoWhisper-large"
    PHOWHISPER_LOCAL_PATH = MODELS_DIR / "phowhisper-large"
    
    # Whisper V2 - Fallback runtime now backed by faster-whisper/CTranslate2
    WHISPER_V2_MODEL = "openai/whisper-large-v2"
    WHISPER_V2_LOCAL_PATH = MODELS_DIR / "whisper-large-v2"
    
    # Word timestamps
    WORD_TIMESTAMPS = True
    
    # Anti-hallucination settings
    CONDITION_ON_PREVIOUS_TEXT = False
    COMPRESSION_RATIO_THRESHOLD = 2.0
    NO_SPEECH_THRESHOLD = 0.5


class DiarizationConfig:
    """Offline diarization parameters."""
    # Engine Selection: "speechbrain" (offline default) or "pyannote" (optional online/authenticated)
    ENGINE = "speechbrain"

    # HuggingFace Token for Pyannote model download
    # Get token at: https://huggingface.co/settings/tokens
    # Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    
    # SpeechBrain legacy config (only used if ENGINE = "speechbrain")
    SEGMENT_DURATION = 1.2
    STEP_DURATION = 0.2
    CLUSTERING_TYPE = "spectral"
    MAX_SPEAKERS = 10

class RefinementConfig:
    """LLM Speaker Refinement Parameters"""
    ENABLED = False # Disabled for robustness (User preference: generic Speaker IDs)
    MODEL_TEMP = 0.1 # Low temp for deterministic logic
    CONTEXT_WINDOW = 2000 # Tokens
    
    # Prompt Template
    PROMPT_TEMPLATE = "role_inference.j2"
    
    # Dynamic Role Inference (No hardcoded roles)
    # The system will deduce roles like "Mother", "Police", "Seller" from context.

