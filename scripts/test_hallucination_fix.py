import sys
import os
import time
import logging

import soundfile as sf
import torch
import torchaudio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.factories.system_factory import SystemFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioLoader:
    """NumPy-2.x-compatible loader without librosa/numba."""

    def load(self, path: str, sr: int = 16000):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(audio)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
            sample_rate = sr

        return waveform.numpy(), sample_rate

def test_hallucination_fix():
    # Correct path to the audio file in the root
    audio_path = "e:/research/Cherry2/Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại.mp3"
    
    # Check if exists
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return
    
    print("\n🧪 TEST: Whisper V3 Hallucination Fix (Rapid Check)")
    print("-" * 50)
    
    # 1. Load Audio Snippet (First 60s where 'ừ ừ' occurs?)
    # Actually the hallucination 'ừ ừ' was at the START.
    loader = AudioLoader()
    audio, sr = loader.load(audio_path)
    
    # Take first 30 seconds (16000 * 30 samples)
    snippet = audio[:16000*30]
    
    print(f"🎵 Audio loaded: {len(snippet)} samples (30s)")
    
    # 2. Load Model
    factory = SystemFactory()
    # Create the internal model directly to verify generate params?
    # Or use the adapter? Adapter uses the model inside.
    
    transcriber = factory.create_transcriber("whisper-v3")
    
    # We need to access the internal 'LegacyTranscriber' -> 'model' to run transcribe_segment manually
    # or just run transcribe() on a temp file?
    
    # Let's save the snippet to temp file and run transcribe on it.
    temp_file = "temp_snippet.wav"
    sf.write(temp_file, snippet, 16000)
    
    print("🚀 Transcribing snippet...")
    start = time.time()
    
    # Use the adapter
    transcript = transcriber.transcribe(temp_file)
    
    end = time.time()
    print(f"✅ Done in {end - start:.2f}s")
    
    print("\n📜 RESULT:")
    print(transcript.text)
    
    # Validation
    if "ừ ừ" in transcript.text:
        print("\n❌ FAIL: 'ừ ừ' detected!")
    else:
        print("\n✅ PASS: No 'ừ ừ' detected.")
        
    if "Subscibe" in transcript.text or "YouTube" in transcript.text:
         print("\n❌ FAIL: YouTube hallucination detected!")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    test_hallucination_fix()
