"""
Download Single VIVOS Sample
"""
import requests
from pathlib import Path

URL = "https://upload.wikimedia.org/wikipedia/commons/0/07/Voa_vietnamese_20161026_0100.mp3"
OUTPUT_PATH = Path("samples/benchmark/vivos/voa_sample.wav") # Reuse vivos folder

def download_file():
    print(f"Downloading VOA sample from {URL}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(URL, stream=True, timeout=30, headers=headers)
        if response.status_code == 200:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            temp_mp3 = OUTPUT_PATH.with_suffix(".mp3")
            with open(temp_mp3, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert and trim using pydub
            from pydub import AudioSegment
            print("Converting and trimming first 30s...")
            audio = AudioSegment.from_file(temp_mp3)
            # Take first 30 seconds
            cut_audio = audio[:30000] 
            cut_audio.export(OUTPUT_PATH, format="wav")
            
            # Clean up
            temp_mp3.unlink()
            
            print(f"✅ Saved to {OUTPUT_PATH} ({len(cut_audio)}ms)")
            
            # Save descriptive transcript (Not ground truth, just marker)
            with open(OUTPUT_PATH.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write("Đây là bản tin VOA tiếng Việt (transcript mẫu)")
                
            return True
        else:
            print(f"❌ Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    download_file()
