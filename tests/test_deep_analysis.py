from application.services.analysis_service import AnalysisService


class FakeEngine:
    def load(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        assert "FutureCoin" in prompt
        assert "lùa gà" in prompt
        return """
```json
{
  "sva_analysis": {
    "criteria_met": ["Ngôn ngữ thuyết phục"],
    "credibility_score": 32,
    "analysis": "Có dấu hiệu dẫn dụ nạn nhân."
  },
  "scan_linguistics": {
    "pronoun_shifts": ["Chúng tôi"],
    "time_gaps": ["Sau đó"],
    "deception_markers": ["Hình như"],
    "conclusion": "Ngôn ngữ thiếu cam kết."
  },
  "psychological_profile_vn": {
    "concealment_tactics": ["Tạo FOMO"],
    "manipulation_type": "Lùa đầu tư",
    "role_assessment": "Người chào mời",
    "risk_level": "HIGH"
  },
  "threat_level": "HIGH",
  "classification": "Gian lận công nghệ"
}
```
""".strip()


def test_deep_analysis_parses_structured_json_from_engine():
    analyzer = AnalysisService(engine=FakeEngine())

    transcript = """
    A (Sale): Alo, chào anh. Chúng tôi bên dự án FutureCoin. Hiện tại bên em đang có kèo x2 tài khoản trong 24h.
    B (Victim): Thật không em? Nghe giống lùa gà thế.
    A: Dạ không anh. Chúng tôi làm ăn uy tín. Hôm qua có anh khách nạp 500 triệu, sau đó rút về 1 tỷ ngon ơ.
    B: Thế nạp vào đâu?
    A: Anh cứ chuyển vào ví lạnh của sàn. Hình như phí gas đang rẻ đấy. Anh vào cổng game của tụi em check thử đi.
    """.strip()

    report = analyzer.analyze_transcript(transcript, scenario="high_tech_fraud")

    assert report["sva_analysis"]["credibility_score"] == 32
    assert "Chúng tôi" in report["scan_linguistics"]["pronoun_shifts"]
    assert report["psychological_profile_vn"]["risk_level"] == "HIGH"
    assert report["classification"] == "Gian lận công nghệ"
