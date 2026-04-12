from __future__ import annotations

import re
from typing import Any


class IntelPresentationService:
    """Build lightweight, offline-friendly intelligence views from report JSON and transcript text."""

    PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+84|0)(?:\d[\s.\-]?){8,10}\d(?!\d)")
    MONEY_PATTERN = re.compile(
        r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:\s*)(?:đồng|vnđ|VND|triệu|tr|nghìn|ngàn|k)?",
        re.IGNORECASE,
    )
    TIME_PATTERN = re.compile(
        r"\b(?:\d{1,2}[:h]\d{2}|\d{1,2}\s*giờ(?:\s*\d{1,2}\s*phút)?|hôm nay|hôm qua|ngày mai|"
        r"sáng nay|chiều nay|tối nay|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
        re.IGNORECASE,
    )
    LOCATION_PATTERN = re.compile(
        r"\b(?:bến xe|nhà ga|sân bay|khách sạn|nhà nghỉ|quán cà phê|quán cafe|quận|huyện|phường|xã|"
        r"thành phố|tỉnh|đường|phố|ngõ|hẻm)\s+[A-ZÀ-Ỵa-zà-ỵ0-9][A-ZÀ-Ỵa-zà-ỵ0-9\s./-]{1,40}",
        re.IGNORECASE,
    )
    PERSON_PATTERN = re.compile(
        r"\b(?:ông|bà|anh|chị|em|cô|chú|bác)\s+[A-ZÀ-Ỵ][a-zà-ỵ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỵ]+){0,3}",
        re.IGNORECASE,
    )
    SLANG_HINTS = {
        "kẹo": "Thuốc lắc hoặc hàng cấm dạng viên",
        "đồ": "Hàng cấm hoặc vật phẩm nhạy cảm",
        "hàng": "Hàng cấm hoặc tang vật",
        "cơm": "Ẩn ngữ chỉ ma túy/heroin trong một số ngữ cảnh",
        "gỗ": "Ẩn ngữ chỉ heroin/bánh heroin",
        "lavie": "Ẩn ngữ có thể chỉ ma túy đá",
        "dưa": "Ẩn ngữ có thể chỉ ma túy đá",
        "xế": "Người vận chuyển",
        "việc nhẹ lương cao": "Dấu hiệu lừa đảo tuyển dụng",
        "otp": "Liên quan mã xác thực ngân hàng",
        "chuyển tiền gấp": "Dấu hiệu thúc ép giao dịch tài chính",
        "lùa gà": "Dẫn dụ nạn nhân vào kèo đầu tư/lừa đảo",
    }

    def build(self, report: dict[str, Any], transcript: str, scenario: str | None = None) -> dict[str, Any]:
        transcript = (transcript or "").strip()
        report = report or {}

        strategic = report.get("strategic_assessment") or {}
        tactical = report.get("tactical_intelligence") or {}
        behavioral = report.get("behavioral_profiling") or {}
        intelligence = tactical.get("intelligence_5w1h") or {}
        quantitative = tactical.get("quantitative_data") or {}
        sensitive_info = tactical.get("sensitive_info") or {}

        people_items = self._extract_people(intelligence, transcript)
        location_items = self._extract_locations(intelligence, transcript)
        sensitive_items = self._extract_sensitive_items(sensitive_info, transcript)
        finance_items = self._extract_financial_items(quantitative, transcript)
        slang_items = self._extract_slang_items(report, transcript)
        timeline_items = self._build_timeline(intelligence, transcript)
        risk_flags = self._build_risk_flags(
            strategic=strategic,
            behavioral=behavioral,
            sensitive_items=sensitive_items,
            slang_items=slang_items,
            report=report,
        )

        cards = [
            self._build_card(
                "subjects",
                "Chủ thể và liên hệ",
                people_items,
                "Các đối tượng, vai trò và mạng quan hệ nổi bật trong hội thoại.",
            ),
            self._build_card(
                "locations",
                "Địa điểm và mốc nhạy cảm",
                location_items,
                "Địa điểm được nhắc đến hoặc có khả năng cần xác minh.",
            ),
            self._build_card(
                "sensitive",
                "Thông tin nhạy cảm",
                sensitive_items,
                "Số điện thoại, tài khoản, định danh hoặc thông tin cần bảo vệ.",
            ),
            self._build_card(
                "financial",
                "Tài chính và giao dịch",
                finance_items,
                "Các con số, khoản tiền và giao dịch xuất hiện trong nội dung.",
            ),
            self._build_card(
                "slang",
                "Từ lóng và ám hiệu",
                slang_items,
                "Các cụm từ cần suy diễn thêm theo bối cảnh điều tra.",
            ),
        ]

        return {
            "intel_cards": [card for card in cards if card["items"]],
            "intel_timeline": timeline_items,
            "risk_flags": risk_flags,
            "scenario": scenario or "general_intelligence",
        }

    def _build_card(self, card_id: str, title: str, items: list[dict[str, str]], summary: str) -> dict[str, Any]:
        return {
            "id": card_id,
            "title": title,
            "summary": summary,
            "items": items[:8],
        }

    def _extract_people(self, intelligence: dict[str, Any], transcript: str) -> list[dict[str, str]]:
        raw_people = intelligence.get("people") or []
        items: list[dict[str, str]] = []

        for person in raw_people:
            if isinstance(person, dict):
                name = self._clean_value(person.get("name")) or "Chưa rõ danh tính"
                role = self._clean_value(person.get("role")) or "Chưa rõ vai trò"
                network = person.get("relationship_network") or []
                meta = ""
                if isinstance(network, list) and network:
                    meta = "Mạng quan hệ: " + ", ".join(str(item) for item in network[:3] if item)
                items.append({"label": name, "value": role, "meta": meta})
            elif person:
                items.append({"label": str(person), "value": "Nhân vật được nhắc tới", "meta": ""})

        if not items:
            for match in self.PERSON_PATTERN.findall(transcript):
                items.append({"label": self._titleize(match), "value": "Nhân vật xuất hiện trong transcript", "meta": ""})

        return self._dedupe_items(items)

    def _extract_locations(self, intelligence: dict[str, Any], transcript: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        events = intelligence.get("events") or []

        for event in events:
            if not isinstance(event, dict):
                continue
            location = self._clean_value(event.get("location"))
            if not location:
                continue
            detail_parts = []
            action = self._clean_value(event.get("action"))
            event_time = self._clean_value(event.get("time"))
            if action:
                detail_parts.append(action)
            if event_time:
                detail_parts.append(event_time)
            items.append(
                {
                    "label": location,
                    "value": "Địa điểm được nhắc đến",
                    "meta": " | ".join(detail_parts),
                }
            )

        if not items:
            for match in self.LOCATION_PATTERN.findall(transcript):
                items.append({"label": self._clean_value(match), "value": "Địa điểm trong transcript", "meta": ""})

        return self._dedupe_items(items)

    def _extract_sensitive_items(self, sensitive_info: Any, transcript: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []

        if isinstance(sensitive_info, dict):
            pii_entries = sensitive_info.get("pii_detected") or []
            for entry in pii_entries:
                if not isinstance(entry, dict):
                    continue
                label = self._clean_value(entry.get("type")) or "PII"
                value = self._clean_value(entry.get("value")) or "Không rõ"
                owner = self._clean_value(entry.get("owner"))
                meta = f"Chủ sở hữu: {owner}" if owner else ""
                items.append({"label": label, "value": value, "meta": meta})

            for secret in sensitive_info.get("secrets") or []:
                if secret:
                    items.append({"label": "Bí mật/chi tiết kín", "value": str(secret), "meta": ""})

            for vulnerability in sensitive_info.get("vulnerabilities") or []:
                if vulnerability:
                    items.append({"label": "Điểm dễ tổn thương", "value": str(vulnerability), "meta": ""})
        elif isinstance(sensitive_info, list):
            for entry in sensitive_info:
                if not isinstance(entry, dict):
                    continue
                label = self._clean_value(entry.get("type") or entry.get("category")) or "Thông tin nhạy cảm"
                value = self._clean_value(entry.get("value")) or "Không rõ"
                owner = self._clean_value(entry.get("owner"))
                meta = f"Chủ thể: {owner}" if owner else ""
                items.append({"label": label, "value": value, "meta": meta})

        for phone in self.PHONE_PATTERN.findall(transcript):
            items.append({"label": "Số điện thoại", "value": self._clean_value(phone), "meta": "Phát hiện bằng regex offline"})

        for email in re.findall(r"[\w.\-]+@[\w.\-]+\.\w+", transcript):
            items.append({"label": "Email", "value": email, "meta": "Phát hiện bằng regex offline"})

        for bank_number in re.findall(r"\b\d{8,16}\b", transcript):
            if len(bank_number) >= 9:
                items.append({"label": "Số định danh/tài khoản", "value": bank_number, "meta": "Cần xác minh ngữ cảnh"})

        return self._dedupe_items(items)

    def _extract_financial_items(self, quantitative: dict[str, Any], transcript: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []

        for entry in quantitative.get("financials") or []:
            if not isinstance(entry, dict):
                continue
            amount = self._clean_value(entry.get("amount")) or "Không rõ"
            currency = self._clean_value(entry.get("currency"))
            context = self._clean_value(entry.get("context"))
            value = amount if not currency else f"{amount} {currency}".strip()
            items.append({"label": "Giao dịch", "value": value, "meta": context or ""})

        for amount in self.MONEY_PATTERN.findall(transcript):
            normalized = self._clean_value(amount)
            digit_count = len(re.sub(r"\D", "", normalized))
            has_money_hint = any(token in normalized.lower() for token in ("đồng", "vnđ", "vnd", "triệu", "tr", "nghìn", "ngàn", "k"))
            if normalized and any(char.isdigit() for char in normalized):
                if digit_count >= 8 and not has_money_hint:
                    continue
                items.append({"label": "Khoản tiền được nhắc", "value": normalized, "meta": "Phát hiện bằng regex offline"})

        return self._dedupe_items(items)

    def _extract_slang_items(self, report: dict[str, Any], transcript: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        transcript_lower = transcript.lower()

        raw_data = report.get("behavioral_profiling") or {}
        for key in ("scan_linguistics", "psychological_profile_vn"):
            section = raw_data.get(key) or {}
            if isinstance(section, dict):
                for value in section.values():
                    if isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, str) and self._contains_slang(entry.lower()):
                                items.append({"label": "Ngữ cảnh đáng chú ý", "value": entry, "meta": "Suy ra từ báo cáo phân tích"})
                    elif isinstance(value, str) and self._contains_slang(value.lower()):
                        items.append({"label": "Ngữ cảnh đáng chú ý", "value": value, "meta": "Suy ra từ báo cáo phân tích"})

        for term, meaning in self.SLANG_HINTS.items():
            if term in transcript_lower:
                items.append({"label": term, "value": meaning, "meta": "Phát hiện trong transcript"})

        return self._dedupe_items(items)

    def _build_timeline(self, intelligence: dict[str, Any], transcript: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        for event in intelligence.get("events") or []:
            if not isinstance(event, dict):
                continue
            actors = event.get("actors") if isinstance(event.get("actors"), list) else []
            items.append(
                {
                    "time": self._clean_value(event.get("time")) or "Chưa rõ thời gian",
                    "title": self._clean_value(event.get("action")) or self._clean_value(event.get("description")) or "Diễn biến được nhắc tới",
                    "location": self._clean_value(event.get("location")) or "",
                    "detail": self._clean_value(event.get("method")) or self._clean_value(event.get("description")) or "",
                    "actors": [self._clean_value(actor) for actor in actors if self._clean_value(actor)],
                }
            )

        if not items:
            sentence_candidates = re.split(r"(?<=[.!?])\s+|\n+", transcript)
            for sentence in sentence_candidates:
                sentence = self._clean_value(sentence)
                if not sentence:
                    continue
                time_match = self.TIME_PATTERN.search(sentence)
                items.append(
                    {
                        "time": self._clean_value(time_match.group(0)) if time_match else "Chưa rõ thời gian",
                        "title": sentence[:120],
                        "location": "",
                        "detail": "",
                        "actors": [],
                    }
                )
                if len(items) >= 6:
                    break

        return self._dedupe_timeline(items[:8])

    def _build_risk_flags(
        self,
        strategic: dict[str, Any],
        behavioral: dict[str, Any],
        sensitive_items: list[dict[str, str]],
        slang_items: list[dict[str, str]],
        report: dict[str, Any],
    ) -> list[dict[str, str]]:
        flags: list[dict[str, str]] = []
        threat_level = str(strategic.get("threat_level") or "UNKNOWN").upper()
        verdict = self._clean_value((strategic.get("final_conclusion") or {}).get("verdict"))
        investigator_note = self._clean_value((strategic.get("final_conclusion") or {}).get("investigator_note"))

        flags.append(
            {
                "level": self._normalize_level(threat_level),
                "label": f"Mức đe dọa hệ thống: {threat_level}",
                "detail": verdict or investigator_note or "Chưa có kết luận chi tiết.",
            }
        )

        psycho = behavioral.get("psychological_profile_vn") or {}
        psycho_risk = self._clean_value(psycho.get("risk_level"))
        if psycho_risk:
            flags.append(
                {
                    "level": self._normalize_level(psycho_risk),
                    "label": "Đánh giá tâm lý/nghi vấn",
                    "detail": psycho_risk,
                }
            )

        if sensitive_items:
            flags.append(
                {
                    "level": "medium",
                    "label": "Phát hiện thông tin nhạy cảm",
                    "detail": f"{len(sensitive_items)} mục có thể cần che chắn hoặc xác minh thêm.",
                }
            )

        if slang_items:
            flags.append(
                {
                    "level": "medium",
                    "label": "Phát hiện từ lóng hoặc ám hiệu",
                    "detail": f"{len(slang_items)} tín hiệu cần suy diễn theo bối cảnh nghiệp vụ.",
                }
            )

        recommendations = report.get("operational_recommendations") or []
        if recommendations:
            flags.append(
                {
                    "level": "low",
                    "label": "Có khuyến nghị tác nghiệp",
                    "detail": str(recommendations[0]),
                }
            )

        return self._dedupe_flags(flags)

    def _dedupe_items(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            label = self._clean_value(item.get("label"))
            value = self._clean_value(item.get("value"))
            if not label or not value:
                continue
            key = (label.casefold(), value.casefold())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "label": label,
                    "value": value,
                    "meta": self._clean_value(item.get("meta")),
                }
            )
        return deduped

    def _dedupe_timeline(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            time = self._clean_value(item.get("time")) or "Chưa rõ thời gian"
            title = self._clean_value(item.get("title"))
            if not title:
                continue
            key = (time.casefold(), title.casefold())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "time": time,
                    "title": title,
                    "location": self._clean_value(item.get("location")),
                    "detail": self._clean_value(item.get("detail")),
                    "actors": item.get("actors") or [],
                }
            )
        return deduped

    def _dedupe_flags(self, flags: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for flag in flags:
            label = self._clean_value(flag.get("label"))
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "level": self._normalize_level(flag.get("level")),
                    "label": label,
                    "detail": self._clean_value(flag.get("detail")),
                }
            )
        return deduped

    def _contains_slang(self, text: str) -> bool:
        return any(term in text for term in self.SLANG_HINTS)

    def _normalize_level(self, value: Any) -> str:
        normalized = str(value or "medium").strip().lower()
        if normalized in {"critical", "khẩn cấp", "very high", "rất cao"}:
            return "critical"
        if normalized in {"high", "cao"}:
            return "high"
        if normalized in {"low", "thấp"}:
            return "low"
        return "medium"

    def _clean_value(self, value: Any) -> str:
        if value is None:
            return ""
        cleaned = re.sub(r"\s+", " ", str(value)).strip(" \n\t-:|")
        return cleaned

    def _titleize(self, value: str) -> str:
        return " ".join(part.capitalize() for part in self._clean_value(value).split())
