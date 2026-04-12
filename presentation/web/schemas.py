from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobStatusView(BaseModel):
    job_id: str
    state: str
    stage: str
    progress: int
    created_at: str
    updated_at: str
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class TranscriptResultView(BaseModel):
    job_id: str
    language: str
    raw_text: str | None = None
    stable_text: str | None = None
    filtered_text: str | None = None
    corrected_text: str | None = None
    intel_summary: str | None = None
    intel_report: dict[str, Any] = Field(default_factory=dict)
    intel_cards: list[dict[str, Any]] = Field(default_factory=list)
    intel_timeline: list[dict[str, Any]] = Field(default_factory=list)
    risk_flags: list[dict[str, Any]] = Field(default_factory=list)
    speaker_transcript: str | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    stable_segments: list[dict[str, Any]] = Field(default_factory=list)
    speaker_segments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    downloads: dict[str, str] = Field(default_factory=dict)
