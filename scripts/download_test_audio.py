"""
Download Vietnamese Test Audio with Transcript

Tải audio test tiếng Việt thực tế với transcript từ các nguồn:
1. VIVOS corpus (HuggingFace) - Clean speech, single speaker
2. Common Voice Vietnamese - Multiple speakers, varied quality
3. Custom YouTube download (nếu cần)

Usage:
    python scripts/download_test_audio.py
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "benchmark"


def setup_directories():
    """Create benchmark sample directories."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "vivos").mkdir(exist_ok=True)
    (SAMPLES_DIR / "commonvoice").mkdir(exist_ok=True)
    logger.info(f"✅ Created sample directories at: {SAMPLES_DIR}")


def download_vivos_samples(num_samples: int = 1):
    """
    Download sample audio from VIVOS corpus via HuggingFace.
    Clean, studio-quality Vietnamese speech.
    """
    logger.info("📥 Downloading VIVOS samples from HuggingFace...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        # Load VIVOS test split with streaming=True to avoid full download
        logger.info("   Mode: Streaming (getting first valid samples)...")
        dataset = load_dataset("vivos", split="test", streaming=True, trust_remote_code=True)
        
        # Save first N samples
        output_dir = SAMPLES_DIR / "vivos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for sample in dataset:
            if count >= num_samples:
                break
            
            try:
                audio = sample.get("audio")
                transcript = sample.get("sentence")
                
                if not audio or not transcript:
                    continue
                    
                # Save audio (audio['array'] is numpy array)
                audio_path = output_dir / f"vivos_{count+1:03d}.wav"
                sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
                
                # Save transcript
                transcript_path = output_dir / f"vivos_{count+1:03d}.txt"
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                
                logger.info(f"   Saved: {audio_path.name}")
                count += 1
            except Exception as e:
                logger.warning(f"   Skipping sample due to error: {e}")
                
        logger.info(f"✅ Downloaded {count} VIVOS samples")
        return count > 0
    except Exception as e:
        logger.error(f"❌ Failed to download VIVOS: {e}")
        return False


def download_commonvoice_samples(num_samples: int = 5):
    """
    Download sample audio from Common Voice Vietnamese.
    Real users, varied quality, some noise.
    """
    logger.info("📥 Downloading Common Voice Vietnamese samples...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        # Load Common Voice Vietnamese (validated split)
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_0", 
            "vi", 
            split="test",
            trust_remote_code=True
        )
        
        output_dir = SAMPLES_DIR / "commonvoice"
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            
            audio = sample["audio"]
            transcript = sample["sentence"]
            
            # Save audio
            audio_path = output_dir / f"cv_{i+1:03d}.wav"
            sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
            
            # Save transcript
            transcript_path = output_dir / f"cv_{i+1:03d}.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            logger.info(f"   Saved: {audio_path.name}")
        
        logger.info(f"✅ Downloaded {min(num_samples, len(dataset))} Common Voice samples")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download Common Voice: {e}")
        logger.error("   You may need to accept the license at huggingface.co")
        return False


def create_multi_speaker_test():
    """
    Create a multi-speaker test file by concatenating samples.
    This simulates a conversation with multiple speakers.
    """
    logger.info("🔧 Creating multi-speaker test file...")
    
    try:
        import numpy as np
        import soundfile as sf
        
        vivos_dir = SAMPLES_DIR / "vivos"
        audio_files = sorted(vivos_dir.glob("*.wav"))[:4]  # Use first 4 files
        
        if len(audio_files) < 2:
            logger.warning("⚠️ Need at least 2 VIVOS samples to create multi-speaker test")
            return False
        
        combined_audio = []
        combined_transcript = []
        
        for i, audio_path in enumerate(audio_files):
            audio, sr = sf.read(str(audio_path))
            
            # Add silence between speakers (1 second)
            if combined_audio:
                silence = np.zeros(sr)  # 1 second silence
                combined_audio.append(silence)
            
            combined_audio.append(audio)
            
            # Load transcript
            transcript_path = audio_path.with_suffix(".txt")
            if transcript_path.exists():
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    combined_transcript.append(f"[SPEAKER_{i+1}]: {text}")
        
        # Save combined audio
        output_audio = SAMPLES_DIR / "multi_speaker_test.wav"
        combined = np.concatenate(combined_audio)
        sf.write(str(output_audio), combined, sr)
        
        # Save combined transcript
        output_transcript = SAMPLES_DIR / "multi_speaker_test.txt"
        with open(output_transcript, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_transcript))
        
        logger.info(f"✅ Created multi-speaker test: {output_audio}")
        logger.info(f"   Duration: {len(combined)/sr:.1f} seconds")
        logger.info(f"   Speakers: {len(audio_files)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create multi-speaker test: {e}")
        return False


def list_available_samples():
    """List all available test samples."""
    logger.info("\n📁 Available test samples:")
    
    for subdir in SAMPLES_DIR.iterdir():
        if subdir.is_dir():
            audio_files = list(subdir.glob("*.wav"))
            logger.info(f"   {subdir.name}/: {len(audio_files)} audio files")
    
    # Root level
    root_audio = list(SAMPLES_DIR.glob("*.wav"))
    if root_audio:
        logger.info(f"   (root): {len(root_audio)} audio files")
        for f in root_audio:
            logger.info(f"      - {f.name}")


def main():
    logger.info("=" * 60)
    logger.info("Vietnamese Test Audio Downloader")
    logger.info("=" * 60)
    
    # 1. Setup directories
    setup_directories()
    
    # 2. Download VIVOS samples (clean, single speaker)
    logger.info("\n" + "=" * 60)
    logger.info("Step 1/3: VIVOS Corpus (Clean speech)")
    logger.info("=" * 60)
    download_vivos_samples(num_samples=5)
    
    # 3. Download Common Voice samples (varied quality)
    logger.info("\n" + "=" * 60)
    logger.info("Step 2/3: Common Voice Vietnamese (Varied quality)")
    logger.info("=" * 60)
    download_commonvoice_samples(num_samples=5)
    
    # 4. Create multi-speaker test
    logger.info("\n" + "=" * 60)
    logger.info("Step 3/3: Create Multi-Speaker Test")
    logger.info("=" * 60)
    create_multi_speaker_test()
    
    # 5. Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    list_available_samples()
    
    logger.info("\n✅ Test audio download complete!")
    logger.info(f"   Samples saved to: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
