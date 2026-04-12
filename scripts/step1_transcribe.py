"""
Step 1: Transcription (ASR)
Runs ASR and saves segments to JSON for diarization.
Supports: whisper-v2 (default), phowhisper (Vietnamese SOTA)
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import argparse

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Step 1: Transcription (ASR)")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument(
        "--engine", 
        type=str, 
        choices=["whisper-v2", "phowhisper"],
        default="whisper-v2",
        help="ASR engine: whisper-v2 (default) or phowhisper (Vietnamese SOTA, WER 4.67%%)"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    print(f"🎤 Transcribing: {audio_path.name}")
    print(f"   Engine: {args.engine}")
    
    factory = SystemFactory()
    
    if args.engine == "phowhisper":
        # Use PhoWhisper (Vietnamese SOTA)
        from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter
        transcriber = PhoWhisperAdapter()
        print("   🇻🇳 Using PhoWhisper (VinAI) - SOTA Vietnamese ASR")
    else:
        # Default: Whisper V2
        transcriber = factory.create_transcriber("whisper-v2")
        print("   Using Whisper V2")
    
    transcript = transcriber.transcribe(str(audio_path))
    
    # Save segments to JSON
    # Naming convention: {audio_stem}_segments.json
    output_json = OUTPUT_DIR / f"{audio_path.stem}_segments.json"
    
    data = {
        "text": transcript.text,
        "segments": transcript.segments,
        "metadata": getattr(transcript, "metadata", {})
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved transcript segments to: {output_json}")
    print(f"   Total text length: {len(transcript.text)} chars")
    print(f"   Total segments: {len(transcript.segments)}")

if __name__ == "__main__":
    main()

