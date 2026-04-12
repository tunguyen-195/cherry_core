from application.services.analysis_service import AnalysisService
from core.ports.llm_port import ILLMEngine
from core.domain.entities import Transcript, StrategicReport

class GenerateStrategicReportUseCase:
    def __init__(self, llm_engine: ILLMEngine):
        self.llm_engine = llm_engine

    def execute(self, transcript: Transcript, scenario: str) -> StrategicReport:
        analysis_service = AnalysisService(engine=self.llm_engine)
        raw_data = analysis_service.analyze_transcript(transcript.text, scenario)

        if "STRATEGIC_ASSESSMENT" in raw_data:
            return StrategicReport(
                strategic_assessment=raw_data.get("STRATEGIC_ASSESSMENT", {}),
                tactical_intelligence=raw_data.get("TACTICAL_INTELLIGENCE", {}),
                behavioral_profiling=raw_data.get("BEHAVIORAL_PROFILING", {}),
                operational_recommendations=raw_data.get("OPERATIONAL_RECOMMENDATIONS", []),
            )

        strategic_assessment = {
            "executive_briefing": raw_data.get("full_summary", raw_data.get("summary", "Không có thông tin")),
            "threat_level": raw_data.get("threat_level", "UNKNOWN"),
            "classification": raw_data.get("classification", "Không có thông tin"),
            "final_conclusion": raw_data.get("final_conclusion", {}),
        }
        tactical_intelligence = {
            "intelligence_5w1h": raw_data.get("intelligence_5w1h", {}),
            "quantitative_data": raw_data.get("quantitative_data", {}),
            "sensitive_info": raw_data.get("sensitive_info", {}),
        }
        behavioral_profiling = {
            "sva_analysis": raw_data.get("sva_analysis", {}),
            "scan_linguistics": raw_data.get("scan_linguistics", {}),
            "psychological_profile_vn": raw_data.get("psychological_profile_vn", {}),
            "emotional_analysis": raw_data.get("emotional_analysis", {}),
        }

        return StrategicReport(
            strategic_assessment=strategic_assessment,
            tactical_intelligence=tactical_intelligence,
            behavioral_profiling=behavioral_profiling,
            operational_recommendations=raw_data.get("recommendations", []),
        )
