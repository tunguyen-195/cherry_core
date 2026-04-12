import sys
import os
import time

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infrastructure.factories.system_factory import SystemFactory
from application.use_cases.transcribe_audio import TranscribeAudioUseCase

def main():
    print("🍒 Cherry Standalone Correction Benchmark")
    print("=" * 50)
    
    file_path = "samples/test_audio.mp3"
    
    factory = SystemFactory()
    transcriber = factory.create_transcriber()
    corrector = factory.create_corrector() # Should load ProtonX/bmd1905
    
    uc = TranscribeAudioUseCase(transcriber, corrector)
    
    print("🔊 Transcribing & Correcting...")
    start = time.time()
    transcript = uc.execute(file_path)
    dur = time.time() - start
    
    original = transcript.metadata.get('original_text', '')
    corrected = transcript.text
    
    print(f"✅ Done in {dur:.2f}s")
    
    output_file = "samples/correction_benchmark.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== ORIGINAL (PhoWhisper) ===\n")
        f.write(original + "\n\n")
        f.write("=== CORRECTED (ProtonX/bmd1905) ===\n")
        f.write(corrected + "\n")
        
    print(f"💾 Saved to {output_file}")
    
    # Simple Diff Preview
    print("\n🔍 Quick Peek (First 200 chars):")
    print(f"ORI: {original[:200]}")
    print(f"COR: {corrected[:200]}")

if __name__ == "__main__":
    main()
