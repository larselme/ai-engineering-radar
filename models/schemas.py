from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, HttpUrl, model_validator


class SourceKind(StrEnum):
    RSS = "rss"
    WEBPAGE = "webpage"


class AnalysisClassification(StrEnum):
    MATERIAL_CHANGE = "material_change"
    EMERGING_SIGNAL = "emerging_signal"
    NOISE = "noise"


class HypeRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JudgeStatus(StrEnum):
    ACCEPT = "accept"
    WATCHLIST = "watchlist"
    REJECT = "reject"
    REVISE = "revise"


class CandidateTerminalStatus(StrEnum):
    ACCEPT = "accept"
    WATCHLIST = "watchlist"
    REJECT = "reject"
    ERROR = "error"


class RunStatus(StrEnum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class SourceConfig(BaseModel):
    name: str
    kind: SourceKind
    url: HttpUrl
    article_url_prefixes: list[str] = Field(default_factory=list)
    include_title_terms: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    id: str
    source_name: str
    title: str
    url: HttpUrl
    published_at: datetime
    content: str
    content_hash: str


class AnalysisResult(BaseModel):
    summary: str
    what_changed: str
    engineering_impact: str
    classification: AnalysisClassification
    confidence: float = Field(ge=0, le=1)


class CritiqueResult(BaseModel):
    objections: list[str]
    unsupported_claims: list[str]
    hype_risk: HypeRisk
    alternative_interpretation: str


class JudgeDecision(BaseModel):
    status: JudgeStatus
    reason: str
    feedback: str = ""
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_feedback_for_revision(self) -> Self:
        if self.status is JudgeStatus.REVISE and not self.feedback.strip():
            raise ValueError("feedback is required when status is revise")
        return self


class CandidateRecord(BaseModel):
    source: SourceItem
    analyses: list[AnalysisResult] = Field(default_factory=list)
    critiques: list[CritiqueResult] = Field(default_factory=list)
    decisions: list[JudgeDecision] = Field(default_factory=list)
    revision_count: int = Field(default=0, ge=0)
    terminal_status: CandidateTerminalStatus | None = None
    error: str | None = None


class EditorFinding(BaseModel):
    title: str
    source_url: HttpUrl
    what_changed: str
    why_it_matters: str
    confidence: float = Field(ge=0, le=1)
    skeptic_objection: str


class EditorReport(BaseModel):
    top_findings: list[EditorFinding] = Field(max_length=5)
    watchlist: list[EditorFinding] = Field(max_length=5)


class RunRecord(BaseModel):
    run_id: str
    started_at: datetime
    ended_at: datetime | None = None
    previous_successful_run: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    source_errors: dict[str, str] = Field(default_factory=dict)
    candidates: dict[str, CandidateRecord] = Field(default_factory=dict)
    accepted_ids: list[str] = Field(default_factory=list)
    watchlist_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    error_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
