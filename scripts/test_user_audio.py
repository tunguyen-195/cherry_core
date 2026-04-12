"""
Direct Test Script for User-Specified Audio
Bypasses CLI to test full pipeline.
"""
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory
from application.services.analysis_service import AnalysisService

# USER AUDIO FILE
AUDIO_FILE = Path(r"E:\research\Cherry2\Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại.mp3")
OUTPUT_FILE = Path(__file__).parent.parent / "samples" / "user_audio_result.json"

def main():
    print(f"🎤 Testing with: {AUDIO_FILE.name}")
    
    if not AUDIO_FILE.exists():
        print(f"❌ File not found: {AUDIO_FILE}")
        return
    
    factory = SystemFactory()
    
    # Step 1: Transcribe
    print("📝 Step 1: Transcribing audio...")
    transcriber = factory.create_transcriber()
    from core.domain.entities import Transcript
    
    try:
        transcript = transcriber.transcribe(str(AUDIO_FILE))
        print(f"✅ Transcription complete: {len(transcript.text)} chars")
        print(f"   Preview: {transcript.text[:200]}...")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return
    
    # Step 2: Correction (Optional)
    print("\n✍️ Step 2: Correcting spelling...")
    try:
        corrector = factory.create_corrector()
        corrected_text = corrector.correct(transcript.text)
        print(f"✅ Correction complete")
    except Exception as e:
        print(f"⚠️ Correction skipped: {e}")
        corrected_text = transcript.text
    
    # Step 3: LLM Analysis
    print("\n🧠 Step 3: Running LLM Analysis...")
    try:
        summarizer = AnalysisService()
        result = summarizer.analyze_transcript(corrected_text, scenario="general_intelligence")
        print(f"✅ Analysis complete")
        
        # Save result
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved to: {OUTPUT_FILE}")
        
        # Print summary
        if "error" in result:
            print(f"⚠️ Result has error: {result.get('error')}")
        else:
            print("🎉 SUCCESS - Valid JSON output!")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
