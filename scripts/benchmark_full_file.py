"""
Benchmark Standard File (test_audio.mp3)
Chạy Full WhisperX Pipeline trên file test chuẩn của user.
Xuất ra Transcript (.txt) và Detailed Report (.json).
"""
import sys
import json
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.adapters.asr.whisperx_adapter import WhisperXAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO_FILE = Path("samples/test_audio.mp3")
OUTPUT_TXT = Path("samples/test_audio_whisperx.txt")
OUTPUT_JSON = Path("samples/test_audio_whisperx.json")

def run():
    if not AUDIO_FILE.exists():
        logger.error(f"❌ Audio file not found: {AUDIO_FILE}")
        return

    logger.info(f"🚀 Starting Benchmark on: {AUDIO_FILE}")
    start_time = time.time()

    try:
        # Init Adapter
        adapter = WhisperXAdapter(
            model_size="large-v2",
            language="vi",
            batch_size=8, # Optimized batch size
            min_speakers=2, # Expected conversation
            max_speakers=5
        )
        
        # Run Pipeline
        logger.info("⏳ Processing... (This may take a minute)")
        segments = adapter.transcribe_and_diarize(str(AUDIO_FILE))
        
        duration = time.time() - start_time
        logger.info(f"✅ Completed in {duration:.2f}s")
        
        # Process Results
        full_text = " ".join([s.text for s in segments])
        speakers = set([s.speaker_id for s in segments])
        
        # Save TXT
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            for s in segments:
                line = f"[{s.start_time:.2f}s - {s.end_time:.2f}s] {s.speaker_id}: {s.text}"
                f.write(line + "\n")
        
        # Save JSON (Compatible with UI)
        result_data = {
            "meta": {
                "file": str(AUDIO_FILE),
                "duration_proc": duration,
                "model": "whisperx-large-v2",
                "timestamp": time.ctime()
            },
            "speakers": list(speakers),
            "transcript": full_text,
            "segments": [
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker_id,
                    "text": s.text,
                    "words": s.words
                }
                for s in segments
            ]
        }
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"📄 Saved Transcript: {OUTPUT_TXT}")
        logger.info(f"📊 Saved JSON Report: {OUTPUT_JSON}")
        logger.info(f"🗣️  Detected Speakers: {speakers}")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    run()
