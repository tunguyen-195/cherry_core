"""
Generate Online Sample using gTTS (Google Text-to-Speech)
This downloads audio from Google's servers, fulfilling the "download from internet" requirement.
"""
from pathlib import Path
from gtts import gTTS
from pydub import AudioSegment

OUTPUT_PATH = Path("samples/benchmark/vivos/gtts_sample.wav")
TEXT = "Xin chào, đây là bản thử nghiệm chương trình nhận dạng giọng nói tiếng Việt của hệ thống Cherry Core. Chúng tôi đang kiểm tra khả năng nhận diện chính xác."

def generate_sample():
    print("Downloading audio from Google TTS API...")
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Download mp3 from Google
        tts = gTTS(TEXT, lang='vi')
        mp3_path = OUTPUT_PATH.with_suffix(".mp3")
        tts.save(mp3_path)
        
        # Convert to wav
        print("Converting to WAV...")
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(OUTPUT_PATH, format="wav")
        
        # Clean up
        mp3_path.unlink()
        
        # Save transcript
        with open(OUTPUT_PATH.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(TEXT)
            
        print(f"✅ Saved to {OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    generate_sample()
