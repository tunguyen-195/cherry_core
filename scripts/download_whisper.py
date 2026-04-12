import os
from huggingface_hub import snapshot_download
from pathlib import Path

# Config
DEST_DIR = Path("E:/research/Cherry2/cherry_core/models/whisper-large-v3")
MODEL_ID = "openai/whisper-large-v3"

def download_model():
    print(f"⬇️ Downloading {MODEL_ID} to {DEST_DIR}...")
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    download_model()
