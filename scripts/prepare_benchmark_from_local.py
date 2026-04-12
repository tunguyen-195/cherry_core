"""
Prepare Benchmark Data from Local File

Trích xuất các đoạn audio mẫu từ file thực tế (test_audio.mp3) để tạo bộ benchmark:
1. Clip 1: Speaker 1 (Customer)
2. Clip 2: Speaker 2 (Receptionist)
3. Clip 3: Conversation (Both)

Usage:
    python scripts/prepare_benchmark_from_local.py samples/test_audio.mp3
"""
import json
import logging
import argparse
from pathlib import Path
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("samples/benchmark/local_extract")

def extract_clips(audio_path: Path, segments_path: Path):
    """Extract clips based on segments."""
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎧 Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        segments = data["segments"]

    # Define clips to extract (Time in seconds)
    # Clip 1: Receptionist intro (0-10s)
    # Clip 2: Customer request (12-20s)
    # Clip 3: Short conversation (100-130s)
    
    clips = [
        {"name": "receptionist_intro", "start": 0, "end": 10},
        {"name": "customer_request", "start": 12, "end": 20},
        {"name": "conversation_exchange", "start": 100, "end": 130}
    ]
    
    for clip in clips:
        start_ms = clip["start"] * 1000
        end_ms = clip["end"] * 1000
        
        extracted = audio[start_ms:end_ms]
        output_path = OUTPUT_DIR / f"{clip['name']}.wav"
        
        extracted.export(output_path, format="wav")
        logger.info(f"✅ Extracted: {output_path} ({len(extracted)/1000:.1f}s)")
        
        # Save transcript for this clip (approximate)
        clip_text = []
        for seg in segments:
            # Check overlap
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Simple overlap check
            if seg_end > clip["start"] and seg_start < clip["end"]:
                clip_text.append(seg["text"])
        
        txt_path = result_path = OUTPUT_DIR / f"{clip['name']}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(clip_text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Source audio file")
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    segments_path = Path("output") / f"{audio_path.stem}_segments.json"
    
    if not segments_path.exists():
        logger.error(f"❌ Segments file missing: {segments_path}")
        logger.info("   Run step1_transcribe.py first!")
        return
    
    extract_clips(audio_path, segments_path)

if __name__ == "__main__":
    main()
