"""
Cherry Core V2 - Full Diarization Pipeline
Produces a complete transcript with speaker labels.
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory
from core.domain.entities import SpeakerSegment

# Configuration
AUDIO_FILE = Path(r"E:\research\Cherry2\Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại.mp3")
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def align_transcript_with_speakers(transcript_segments, speaker_segments):
    """
    Align ASR segments with speaker diarization segments.
    Returns a list of (speaker_id, text) tuples.
    """
    aligned = []
    
    for t_seg in transcript_segments:
        # Find the speaker segment that overlaps with this transcript segment
        t_start = t_seg.get("start", 0)
        t_end = t_seg.get("end", 0)
        t_text = t_seg.get("text", "")
        
        if not t_text.strip():
            continue
        
        # Find best matching speaker (by overlap)
        best_speaker = "UNKNOWN"
        best_overlap = 0
        
        for s_seg in speaker_segments:
            s_start = s_seg.start_time
            s_end = s_seg.end_time
            
            # Calculate overlap
            overlap_start = max(t_start, s_start)
            overlap_end = min(t_end, s_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s_seg.speaker_id
        
        aligned.append({
            "start": t_start,
            "end": t_end,
            "speaker": best_speaker,
            "text": t_text
        })
    
    return aligned

def format_diarized_transcript(aligned_segments):
    """
    Format aligned segments into a readable transcript.
    """
    lines = []
    current_speaker = None
    current_text = []
    
    for seg in aligned_segments:
        if seg["speaker"] != current_speaker:
            # Save previous speaker's text
            if current_speaker and current_text:
                lines.append(f"[{current_speaker}]: {' '.join(current_text)}")
            # Start new speaker
            current_speaker = seg["speaker"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])
    
    # Don't forget last segment
    if current_speaker and current_text:
        lines.append(f"[{current_speaker}]: {' '.join(current_text)}")
    
    return "\n\n".join(lines)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    base_name = AUDIO_FILE.stem
    
    print("=" * 60)
    print("🍒 CHERRY CORE V2 - FULL DIARIZATION PIPELINE")
    print("=" * 60)
    print(f"🎤 Audio: {AUDIO_FILE.name}")
    print("=" * 60)
    
    if not AUDIO_FILE.exists():
        print(f"❌ File not found: {AUDIO_FILE}")
        return
    
    factory = SystemFactory()
    
    # ==== STEP 1: TRANSCRIPTION ====
    print("\n📝 STEP 1: TRANSCRIPTION")
    print("-" * 40)
    
    transcriber = factory.create_transcriber("whisper-v2")
    transcript = transcriber.transcribe(str(AUDIO_FILE))
    
    print(f"✅ Transcription complete: {len(transcript.text)} chars")
    print(f"   Segments: {len(transcript.segments)}")
    
    # FREE MEMORY: Unload Whisper before Diarization
    del transcriber
    import torch
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Freed GPU memory.")
    
    # ==== STEP 2: DIARIZATION (ENHANCED) ====
    print("\n🎤 STEP 2: SPEAKER DIARIZATION (Enhanced)")
    print("-" * 40)
    
    try:
        # Use enhanced diarizer with VAD preprocessing and auto speaker detection
        diarizer = factory.create_diarizer(mode="enhanced", n_speakers=None)
        speaker_segments = diarizer.diarize(str(AUDIO_FILE))
        
        # Count unique speakers
        unique_speakers = len(set(seg.speaker_id for seg in speaker_segments))
        print(f"✅ Diarization complete: {len(speaker_segments)} segments, {unique_speakers} speakers")
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return
    
    # ==== STEP 3: ALIGNMENT ====
    print("\n🔗 STEP 3: TRANSCRIPT-SPEAKER ALIGNMENT")
    print("-" * 40)
    
    aligned = align_transcript_with_speakers(transcript.segments, speaker_segments)
    print(f"✅ Aligned {len(aligned)} segments")
    
    # ==== STEP 4: FORMAT & SAVE ====
    print("\n💾 STEP 4: SAVING OUTPUT")
    print("-" * 40)
    
    # Format readable transcript
    formatted = format_diarized_transcript(aligned)
    
    # Save formatted transcript
    output_file = OUTPUT_DIR / f"{base_name}_diarized.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Diarized Transcript\n")
        f.write(f"# Audio: {AUDIO_FILE.name}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Speakers: 2 (SPEAKER_1: Lễ tân, SPEAKER_2: Khách hàng)\n")
        f.write("=" * 60 + "\n\n")
        f.write(formatted)
    
    print(f"✅ Saved: {output_file}")
    
    # Save detailed JSON
    json_file = OUTPUT_DIR / f"{base_name}_diarized.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "audio_file": AUDIO_FILE.name,
                "processed_at": datetime.now().isoformat(),
                "total_segments": len(aligned),
                "speakers": list(set(s["speaker"] for s in aligned))
            },
            "segments": aligned,
            "formatted_transcript": formatted
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved: {json_file}")
    
    # Preview
    print("\n" + "=" * 60)
    print("📜 PREVIEW (First 500 chars):")
    print("=" * 60)
    print(formatted[:500] + "...")

if __name__ == "__main__":
    main()
