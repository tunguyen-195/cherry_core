"""
Verify WhisperX Integration

Script chạy benchmark WhisperX trên tập dữ liệu local đã trích xuất.
So sánh kết quả WhisperX (SOTA) với Transcript gốc (Ground Truth - approximate).

Usage:
    python scripts/verify_whisperx.py
"""
import sys
import logging
from pathlib import Path
from jiwer import wer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.adapters.asr.whisperx_adapter import WhisperXAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path("samples/benchmark/vivos")

def run_benchmark():
    if not BENCHMARK_DIR.exists():
        logger.error(f"❌ Benchmark directory missing: {BENCHMARK_DIR}")
        logger.info("   Run prepare_benchmark_from_local.py first!")
        return

    logger.info("🚀 Initializing WhisperX Adapter...")
    try:
        adapter = WhisperXAdapter(
            model_size="large-v2",
            language="vi",
            batch_size=4
        )
    except Exception as e:
        logger.error(f"❌ Failed to init WhisperX: {e}")
        return

    logger.info("\n📊 Starting Benchmark...")
    logger.info(f"{'Filename':<25} | {'Speakers':<8} | {'WER':<8} | {'Status':<10}")
    logger.info("-" * 65)
    
    reports = []
    
    for audio_file in BENCHMARK_DIR.glob("*.wav"):
        txt_file = audio_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
            
        with open(txt_file, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()
            
        try:
            # Run WhisperX Pipeline
            result_segments = adapter.transcribe_and_diarize(str(audio_file))
            
            # Combine text
            pred_text = " ".join([seg.text for seg in result_segments]).strip()
            
            # Count speakers
            speakers = set([seg.speaker_id for seg in result_segments])
            num_speakers = len(speakers)
            
            # Calculate WER (Word Error Rate)
            if ground_truth:
                error_rate = wer(ground_truth, pred_text)
            else:
                error_rate = 0.0
                
            status = "✅ PASS" if error_rate < 0.2 else "⚠️ HIGH WER"
            
            logger.info(f"{audio_file.name:<25} | {num_speakers:<8} | {error_rate:.2f}     | {status:<10}")
            
            reports.append({
                "file": audio_file.name,
                "speakers": list(speakers),
                "wer": error_rate,
                "pred": pred_text,
                "truth": ground_truth
            })
            
        except Exception as e:
            logger.error(f"{audio_file.name:<25} | ERROR    | {e}")

    # Detailed Report
    logger.info("\n📝 Detailed Analysis:")
    for r in reports:
        logger.info(f"\nExample: {r['file']}")
        logger.info(f"   Truth: {r['truth'][:100]}...")
        logger.info(f"   Pred : {r['pred'][:100]}...")
        logger.info(f"   Speakers: {r['speakers']}")
        logger.info(f"   WER: {r['wer']:.2%}")

    adapter.cleanup()

if __name__ == "__main__":
    run_benchmark()
