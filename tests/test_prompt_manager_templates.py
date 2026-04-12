from application.services.prompt_manager import PromptManager


def test_prompt_manager_renders_forensic_brief_template():
    manager = PromptManager()

    rendered = manager.render_template(
        "forensic_brief.j2",
        scenario_label="Trinh sát tổng hợp",
        threat_level="MEDIUM",
        classification="Theo dõi đối tượng",
        executive_briefing="Đối tượng hẹn gặp và nhắc tới khoản tiền cần giao.",
        verdict="Cần theo dõi",
        investigator_note="Ưu tiên xác minh địa điểm gặp.",
        risk_flags=[{"level": "medium", "label": "Phát hiện thông tin nhạy cảm", "detail": "Có số điện thoại."}],
        subject_items=[{"label": "Nguyễn Văn A", "value": "Người bị theo dõi", "meta": ""}],
        location_items=[{"label": "bến xe Mỹ Đình", "value": "Địa điểm hẹn gặp", "meta": ""}],
        sensitive_items=[{"label": "Số điện thoại", "value": "0912345678", "meta": ""}],
        finance_items=[{"label": "Khoản tiền được nhắc", "value": "5 triệu", "meta": ""}],
        slang_items=[{"label": "kẹo", "value": "Hàng cấm dạng viên", "meta": ""}],
        timeline_items=[{"time": "19 giờ", "title": "gặp mặt", "location": "bến xe Mỹ Đình", "detail": "", "actors": ["Nguyễn Văn A"]}],
        recommendations=["Theo dõi thêm 24 giờ."],
    )

    assert "Kịch bản nghiệp vụ: Trinh sát tổng hợp" in rendered
    assert "Nguyễn Văn A" in rendered
    assert "19 giờ: gặp mặt" in rendered
    assert "Theo dõi thêm 24 giờ." in rendered
