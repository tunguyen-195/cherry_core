import argparse
import sys
import os
import json
import logging

# Add project root to path
# We are in presentation/cli, so root is ../../
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from infrastructure.factories.system_factory import SystemFactory
from application.use_cases.transcribe_audio import TranscribeAudioUseCase
from application.use_cases.generate_report import GenerateStrategicReportUseCase
import core.config as config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    print("🍒 Cherry Core V2 - Modern Forensic System")
    print("=" * 50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Audio file path")
    parser.add_argument(
        "--model",
        choices=["phowhisper", "whisper-v3", "whisper-v2"],
        default=config.ASRConfig.ENGINE,
        help="ASR model backend",
    )
    parser.add_argument("--analyze", action="store_true", help="Generate Strategic Report")
    parser.add_argument("--correct", action="store_true", help="Enable Post-OCR Spell Correction (ProtonX)")
    parser.add_argument("--scenario", default="general_intelligence", help="Intelligence Scenario")
    args = parser.parse_args()

    # 1. Bootstrap
    factory = SystemFactory()
    transcriber = factory.create_transcriber(args.model)
    
    corrector = None
    if args.correct:
        print("\n✨ Enabling ProtonX Spell Correction...")
        corrector = factory.create_corrector()

    # 2. Transcription
    print(f"\n🔊 [1/2] Executing Transcription UseCase...")
    uc_transcribe = TranscribeAudioUseCase(transcriber, corrector)
    transcript = uc_transcribe.execute(args.file)
    
    print("\n📝 Transcript Result:")
    print("-" * 30)
    if transcript.metadata.get('corrected'):
        print(f"🔴 [ORIGINAL]:\n{transcript.metadata.get('original_text')}\n")
        print(f"🟢 [PROTONX CORRECTED]:\n{transcript.text}")
    else:
        print(transcript.text)
    print("-" * 30)

    # 3. Analysis
    if args.analyze:
        llm_engine = factory.create_llm_engine()
        print(f"\n🕵️ [2/2] Executing Intelligence Analysis UseCase (Scenario: {args.scenario})...")
        uc_report = GenerateStrategicReportUseCase(llm_engine)
        report = uc_report.execute(transcript, args.scenario)

        print("\n🚨 STRATEGIC DOSSIER (Summary):")
        print("=" * 60)
        # Access safely via domain object
        brief = report.strategic_assessment.get('executive_briefing', 'N/A')
        threat_level = report.strategic_assessment.get('threat_level', 'UNKNOWN')
        print(f"📌 EXECUTIVE BRIEFING: {brief}")
        print(f"⚠️ THREAT LEVEL: {threat_level}")
        print("=" * 60)
        
        # Save JSON
        report_path = args.file + ".strategic_v2.json"
        with open(report_path, "w", encoding="utf-8") as f:
            # We need to serialize the dataclass. For now, we reconstruct the dict or use asdict
            # Simple approach: Reconstruct dict matching the entities
            data = {
                "STRATEGIC_ASSESSMENT": report.strategic_assessment,
                "TACTICAL_INTELLIGENCE": report.tactical_intelligence,
                "BEHAVIORAL_PROFILING": report.behavioral_profiling,
                "OPERATIONAL_RECOMMENDATIONS": report.operational_recommendations
            }
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Strategic Report saved to: {report_path}")

if __name__ == "__main__":
    main()
