"""Test script for Enhanced Diarizer."""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from infrastructure.factories.system_factory import SystemFactory
from pathlib import Path

# Get test audio
audio_dir = Path("samples")
audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

if not audio_files:
    print("❌ No audio files in data/ directory")
    exit(1)

audio_path = str(audio_files[0])
print(f"Testing on: {audio_path}")

# Create enhanced diarizer (default mode)
factory = SystemFactory()
diarizer = factory.create_diarizer(mode="enhanced")

print(f"Diarizer type: {type(diarizer).__name__}")

# Run diarization
segments = diarizer.diarize(audio_path)

print(f"\n📊 Results:")
print(f"Total segments: {len(segments)}")

# Count speakers
speakers = set(s.speaker_id for s in segments)
print(f"Unique speakers: {len(speakers)}")

# Show first 10 segments
print(f"\nFirst 10 segments:")
for i, seg in enumerate(segments[:10]):
    print(f"  [{seg.start_time:.2f}s - {seg.end_time:.2f}s] {seg.speaker_id}")
