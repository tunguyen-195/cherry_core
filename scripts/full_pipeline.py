"""
Cherry Core V2 - Full Analysis Pipeline
Produces 3 outputs:
1. transcript_raw.txt - Original ASR output
2. transcript_corrected.txt - After spell correction
3. analysis_report.json - Comprehensive investigative analysis
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
# from src.services.summarizer import SummarizationService (Migrated to application/services)

# Configuration
AUDIO_FILE = Path(r"E:\research\Cherry2\Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại.mp3")
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

def save_transcript(text: str, filename: str, output_dir: Path):
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"💾 Saved: {filepath}")
    return filepath

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = AUDIO_FILE.stem
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("🍒 CHERRY CORE V2 - FULL ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"🎤 Input: {AUDIO_FILE.name}")
    print(f"📁 Output Dir: {output_dir}")
    print("=" * 60)
    
    if not AUDIO_FILE.exists():
        print(f"❌ File not found: {AUDIO_FILE}")
        return
    
    factory = SystemFactory()
    
    # ========================================
    # STEP 1: TRANSCRIPTION (ASR)
    # ========================================
    print("\n📝 STEP 1: TRANSCRIPTION (ASR)")
    print("-" * 40)
    
    # Simple CLI for model selection
    model_name = "whisper-v2"
    if len(sys.argv) > 1 and sys.argv[1] in ["phowhisper", "whisper-v3", "whisper-v2"]:
        model_name = sys.argv[1]
    
    print(f"🤖 Model: {model_name}")
    
    transcriber = factory.create_transcriber(model_name=model_name)
    
    try:
        transcript = transcriber.transcribe(str(AUDIO_FILE))
        raw_text = transcript.text
        
        if not raw_text or len(raw_text.strip()) == 0:
            print("⚠️ WARNING: Transcription returned empty. Debugging...")
            print(f"   Transcript object: {transcript}")
            print(f"   Transcript.text type: {type(transcript.text)}")
            print(f"   Transcript.metadata: {transcript.metadata}")
            # Try alternative approach
            raw_text = "(Transcription returned empty - check audio file and model)"
        else:
            print(f"✅ Transcription complete: {len(raw_text)} characters")
            print(f"📜 Preview: {raw_text[:300]}...")
        
        # Save raw transcript
        raw_file = save_transcript(
            raw_text, 
            f"{base_name}_transcript_raw.txt", 
            output_dir
        )
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 2: SPELL CORRECTION
    # ========================================
    print("\n✍️ STEP 2: SPELL CORRECTION")
    print("-" * 40)
    
    try:
        corrector = factory.create_corrector()
        corrected_text = corrector.correct(raw_text)
        print(f"✅ Correction complete: {len(corrected_text)} characters")
        
        # Show differences
        if corrected_text != raw_text:
            print("📝 Changes detected (sample):")
            # Simple diff preview
            raw_words = set(raw_text.lower().split())
            corrected_words = set(corrected_text.lower().split())
            new_words = corrected_words - raw_words
            if new_words:
                print(f"   New/changed words: {list(new_words)[:5]}...")
        else:
            print("📝 No corrections needed")
        
        # Save corrected transcript
        corrected_file = save_transcript(
            corrected_text, 
            f"{base_name}_transcript_corrected.txt", 
            output_dir
        )
    except Exception as e:
        print(f"⚠️ Correction failed: {e}")
        corrected_text = raw_text
        corrected_file = raw_file
    
    # ========================================
    # STEP 2.5: LLM CONTEXTUAL CORRECTION (Generalized)
    # ========================================
    print("\n🧠 STEP 2.5: LLM CONTEXTUAL CORRECTION")
    print("-" * 40)
    
    try:
        from application.services.correction_service import CorrectionService
        llm_corrector = CorrectionService()
        final_text = llm_corrector.correct(corrected_text)
        print(f"✅ LLM Correction complete: {len(final_text)} characters")
        
        if final_text != corrected_text:
            print("📝 Contextual corrections applied (phonetic/semantic fixes)")
        else:
            print("📝 No additional corrections needed")
            
        # Update corrected_text for analysis
        corrected_text = final_text
        
        # Save LLM-corrected transcript
        corrected_file = save_transcript(
            corrected_text, 
            f"{base_name}_transcript_corrected.txt", 
            output_dir
        )
    except Exception as e:
        print(f"⚠️ LLM Correction skipped: {e}")
        # Continue with ProtonX-corrected text
    
    # ========================================
    # STEP 3: COMPREHENSIVE ANALYSIS
    # ========================================
    print("\n🧠 STEP 3: COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    
    try:
        from application.services.analysis_service import AnalysisService
        summarizer = AnalysisService()
        
        # Analyze BOTH raw and corrected for comparison
        print("Analyzing corrected transcript...")
        result_corrected = summarizer.analyze_transcript(
            corrected_text, 
            scenario="general_intelligence"
        )
        
        # Build comprehensive report
        report = {
            "metadata": {
                "source_file": AUDIO_FILE.name,
                "processed_at": datetime.now().isoformat(),
                "transcript_raw_file": str(raw_file),
                "transcript_corrected_file": str(corrected_file),
                "raw_length_chars": len(raw_text),
                "corrected_length_chars": len(corrected_text),
            },
            "full_transcript": corrected_text,
            "summary": result_corrected.get("executive_summary", ""),
            "threat_assessment": {
                "level": result_corrected.get("threat_level", "UNKNOWN"),
                "classification": result_corrected.get("classification", ""),
            },
            "extracted_entities": result_corrected.get("key_entities", {}),
            "timeline": result_corrected.get("timeline", ""),
            "behavioral_analysis": result_corrected.get("behavioral_analysis", ""),
            "intent_analysis": result_corrected.get("intent_analysis", ""),
            "recommendations": result_corrected.get("recommendations", []),
            "raw_analysis": result_corrected,
        }
        
        # Save analysis report
        report_file = output_dir / f"{base_name}_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {report_file}")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("📊 PIPELINE COMPLETE - OUTPUT FILES:")
    print("=" * 60)
    print(f"1. 📄 Raw Transcript:       {raw_file.name}")
    print(f"2. 📄 Corrected Transcript: {corrected_file.name}")
    print(f"3. 📊 Analysis Report:      {report_file.name}")
    print("=" * 60)
    
    # Quick preview of analysis
    print("\n📋 QUICK ANALYSIS PREVIEW:")
    print(f"   Threat Level: {report['threat_assessment']['level']}")
    print(f"   Classification: {report['threat_assessment']['classification']}")
    print(f"   Summary: {report['summary'][:200]}...")

if __name__ == "__main__":
    main()
