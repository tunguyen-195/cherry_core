import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
import logging
import functools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = os.path.abspath("models/phowhisper")
OUTPUT_PATH = os.path.abspath("models/phowhisper-safe")

def unsafe_load_patch():
    """Context manager or patch to force weights_only=False"""
    # Patch torch.load directly
    _original_load = torch.load
    
    def _unsafe_load(*args, **kwargs):
        if 'weights_only' in kwargs:
             kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
        
    torch.load = _unsafe_load
    return _original_load

def restore_load(original_load):
    torch.load = original_load

def convert():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Input model path not found: {MODEL_PATH}")
        return

    logger.info(f"🔄 Loading PhoWhisper from {MODEL_PATH} (Unsafe Mode)...")
    
    # Apply Patch
    original_load = unsafe_load_patch()
    
    try:
        # Load Config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        
        # Load State Dict Manually (Bypassing AutoModel checks)
        model_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
        if not os.path.exists(model_file):
            logger.error(f"❌ pytorch_model.bin not found at {model_file}")
            return

        logger.info(f"🔓 Manually loading state_dict from {model_file}...")
        
        # Apply Patch for torch.load
        original_load = unsafe_load_patch()
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        restore_load(original_load)
        
        # Initialize Model structure
        model = AutoModelForSpeechSeq2Seq.from_config(config)
        
        # Load weights
        model.load_state_dict(state_dict)
        
        # Load Processor / Tokenizer
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        
        logger.info("✅ Model loaded successfully. Saving as SafeTensors...")
        
        # Save safe version
        model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
        processor.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        
        logger.info(f"🎉 Conversion Complete! Model saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"❌ Failed to convert: {e}")
        import traceback
        traceback.print_exc()
    finally:
        restore_load(original_load)

if __name__ == "__main__":
    convert()
