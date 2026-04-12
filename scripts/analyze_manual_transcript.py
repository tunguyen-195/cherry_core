import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory
from application.services.correction_service import CorrectionService
from application.services.analysis_service import AnalysisService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# TARGET FILE
RAW_TRANSCRIPT_PATH = Path(r"E:\research\Cherry2\cherry_core\output\Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại_transcript_raw.txt")
OUTPUT_DIR = RAW_TRANSCRIPT_PATH.parent

def main():
    if not RAW_TRANSCRIPT_PATH.exists():
        print(f"❌ File not found: {RAW_TRANSCRIPT_PATH}")
        return

    print("=" * 60)
    print("🍒 CHERRY CORE V2 - MANUAL ANALYSIS PIPELINE")
    print("=" * 60)
    
    # 1. Read Raw Transcript
    print(f"📖 Reading: {RAW_TRANSCRIPT_PATH.name}")
    with open(RAW_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"   Length: {len(raw_text)} chars")

    # 2. Correction (ProtonX)
    print("\n✍️ STEP 2: SPELL CORRECTION (ProtonX)")
    factory = SystemFactory()
    try:
        corrector = factory.create_corrector()
        corrected_text = corrector.correct(raw_text)
        print(f"✅ Correction complete")
    except Exception as e:
        print(f"⚠️ Correction failed: {e}")
        corrected_text = raw_text

    # 2.5 LLM Correction
    print("\n🧠 STEP 2.5: LLM CONTEXTUAL CORRECTION")
    try:
        llm_corrector = CorrectionService()
        final_text = llm_corrector.correct(corrected_text)
        print(f"✅ LLM Correction complete")
        corrected_text = final_text
    except Exception as e:
        print(f"⚠️ LLM Correction failed: {e}")

    # Save Corrected
    corrected_path = OUTPUT_DIR / str(RAW_TRANSCRIPT_PATH.name).replace("_raw.txt", "_corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)
    print(f"💾 Saved Corrected: {corrected_path.name}")

    # 3. Analysis
    print("\n🧠 STEP 3: COMPREHENSIVE ANALYSIS")
    try:
        summarizer = AnalysisService()
        result = summarizer.analyze_transcript(corrected_text, scenario="general_intelligence")
        
        # Save Report
        report_path = OUTPUT_DIR / str(RAW_TRANSCRIPT_PATH.name).replace("_transcript_raw.txt", "_analysis_report.json")
        
        # Wrap in full structure
        full_report = {
            "metadata": {
                 "source_file": "Manual Analysis",
                 "processed_at": datetime.now().isoformat(),
                 "raw_length": len(raw_text),
                 "final_length": len(corrected_text)
            },
            "full_transcript": corrected_text,
            "raw_analysis": result,
            "summary": result.get("executive_summary", ""),
            "threat_assessment": {
                "level": result.get("threat_level", "UNKNOWN"),
                "classification": result.get("classification", "")
            }
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saved Report: {report_path.name}")
        print("\n✅ Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
