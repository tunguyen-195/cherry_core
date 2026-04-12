"""
Step 2: Diarization & Alignment
Loads transcript segments, runs Enhanced Diarization, and aligns speakers.
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import argparse

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory
from core.domain.entities import SpeakerSegment

OUTPUT_DIR = Path(__file__).parent.parent / "output"

def align_transcript_with_speakers(transcript_segments, speaker_segments):
    aligned = []
    for t_seg in transcript_segments:
        t_start = t_seg.get("start", 0)
        t_end = t_seg.get("end", 0)
        t_text = t_seg.get("text", "")
        if not t_text.strip(): continue
        
        best_speaker = "UNKNOWN"
        best_overlap = 0
        for s_seg in speaker_segments:
            overlap = max(0, min(t_end, s_seg.end_time) - max(t_start, s_seg.start_time))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s_seg.speaker_id
        
        aligned.append({
            "start": t_start, "end": t_end, "speaker": best_speaker, "text": t_text
        })
    return aligned



def main():
    parser = argparse.ArgumentParser(description="Step 2: Diarization & Alignment")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument("--speakers", type=int, default=None, help="Force number of speakers")
    parser.add_argument("--refine", action="store_true", help="Enable LLM-based contextual speaker refinement")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    # Infer transcript JSON path
    transcript_json = OUTPUT_DIR / f"{audio_path.stem}_segments.json"
    
    if not transcript_json.exists():
        print(f"❌ Transcript JSON not found: {transcript_json}")
        print(f"   Please run step1_transcribe.py first.")
        return

    print(f"🎤 Diarizing: {audio_path.name}")
    factory = SystemFactory()
    
    # Use SOTA Diarizer (SpeechBrain)
    try:
        diarizer = factory.create_diarizer(mode="sota", n_speakers=args.speakers)
        speaker_segments = diarizer.diarize(str(audio_path))
        print(f"✅ Diarization complete: {len(speaker_segments)} segments")
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return
    
    # Load transcript segments
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        transcript_segments = data["segments"]
    
    # Align (Word-Level SOTA)
    from core.services.alignment_service import AlignmentService
    print("🔗 Running Word-Level Alignment...")
    aligned = AlignmentService.align_words(transcript_segments, speaker_segments)
    
    # --- Contextual Refinement (LLM) ---
    # Only run if requested via CLI (User preference: Not default)
    if args.refine:
        from core.config import RefinementConfig
        RefinementConfig.ENABLED = True # Override config
        
        print("🧠 Running Contextual Speaker Refinement (LLM)...")
        try:
            # Convert dicts -> SpeakerSegment for Refiner
            refiner_segments = []
            for s in aligned:
                refiner_segments.append(SpeakerSegment(
                    start_time=s["start"], 
                    end_time=s["end"], 
                    speaker_id=s["speaker"],
                    text=s["text"],
                    words=s.get("words", [])
                ))
            
            refiner = factory.create_speaker_refiner()
            refined_segments = refiner.refine(refiner_segments)
            
            # Convert SpeakerSegment -> dicts for Formatter
            aligned = [] # Overwrite variable
            for s in refined_segments:
                aligned.append({
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker_id,
                    "text": s.text
                })
                
        except Exception as e:
            print(f"⚠️ Refinement failed, using raw diarization. Error: {e}")
            # 'aligned' remains as is if failure
    else:
        print("⏩ Skipping Contextual Refinement (Use --refine to enable).")
    
    # --- Post-Processing Pipeline ---
    print("🔧 Applying Vietnamese Post-Processing...")
    
    # 1. Hallucination Filter (BoH + Delooping)
    try:
        from infrastructure.adapters.asr.hallucination_filter import HallucinationFilter
        for seg in aligned:
            seg["text"] = HallucinationFilter.filter(seg["text"], language="vi")
        print("   ✅ Hallucination filter applied")
    except ImportError:
        print("   ⚠️ Hallucination filter not available")
    
    # 2. Vietnamese Domain Corrections
    try:
        from infrastructure.adapters.correction.vietnamese_postprocessor import VietnamesePostProcessor
        processor = VietnamesePostProcessor(domain="hotel")
        for seg in aligned:
            seg["text"] = processor.process(seg["text"])
        print("   ✅ Vietnamese post-processor applied (hotel domain)")
    except ImportError:
        print("   ⚠️ Vietnamese post-processor not available")
    
    # Format Output using Modular Service
    from core.services.output_formatter import OutputFormatter
    formatted = OutputFormatter.format_subtitle_style(aligned)
    
    # Save Output
    base_name = audio_path.stem
    output_file = OUTPUT_DIR / f"{base_name}_diarized_final.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Diarized Transcript (Pyannote + Word-Align + Post-Processing)\n")
        f.write("="*60 + "\n\n")
        f.write(formatted)
        
    print(f"✅ Saved final output: {output_file}")

if __name__ == "__main__":
    main()

