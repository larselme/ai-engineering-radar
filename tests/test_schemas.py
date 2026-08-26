from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from models.schemas import (
    AnalysisClassification,
    AnalysisResult,
    CandidateRecord,
    EditorFinding,
    EditorReport,
    JudgeDecision,
    SourceItem,
)


def test_valid_analysis_result() -> None:
    result = AnalysisResult(
        summary="A concise summary",
        what_changed="A model was released",
        engineering_impact="Lower inference latency",
        classification=AnalysisClassification.MATERIAL_CHANGE,
        confidence=0.8,
    )

    assert result.confidence == 0.8
    assert result.classification is AnalysisClassification.MATERIAL_CHANGE


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_analysis_result_rejects_confidence_outside_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(
            summary="Summary",
            what_changed="Change",
            engineering_impact="Impact",
            classification="material_change",
            confidence=confidence,
        )


@pytest.mark.parametrize("status", ["accept", "watchlist", "reject", "revise"])
def test_judge_decision_accepts_each_allowed_status(status: str) -> None:
    feedback = "Reassess the evidence" if status == "revise" else ""

    decision = JudgeDecision(
        status=status,
        reason="Reason",
        feedback=feedback,
        confidence=0.5,
    )

    assert decision.status.value == status


def test_judge_decision_rejects_revise_without_feedback() -> None:
    with pytest.raises(ValidationError):
        JudgeDecision(
            status="revise",
            reason="More analysis is needed",
            confidence=0.5,
        )


def test_candidate_record_rejects_negative_revision_count() -> None:
    source = SourceItem(
        id="item-1",
        source_name="Official source",
        title="Announcement",
        url="https://example.com/announcement",
        published_at=datetime(2026, 8, 25, tzinfo=UTC),
        content="Announcement body",
        content_hash="sha256-value",
    )

    with pytest.raises(ValidationError):
        CandidateRecord(source=source, revision_count=-1)


def test_editor_report_rejects_more_than_five_top_findings() -> None:
    finding = EditorFinding(
        title="Finding",
        source_url="https://example.com/finding",
        what_changed="Something changed",
        why_it_matters="It affects engineers",
        confidence=0.7,
        skeptic_objection="Evidence is preliminary",
    )

    with pytest.raises(ValidationError):
        EditorReport(top_findings=[finding] * 6, watchlist=[])


def test_mutable_defaults_are_not_shared() -> None:
    first = EditorReport(top_findings=[], watchlist=[])
    second = EditorReport(top_findings=[], watchlist=[])

    first.top_findings.append(
        EditorFinding(
            title="Finding",
            source_url="https://example.com/finding",
            what_changed="Something changed",
            why_it_matters="It affects engineers",
            confidence=0.7,
            skeptic_objection="Evidence is preliminary",
        )
    )

    assert second.top_findings == []


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_editor_finding_rejects_confidence_outside_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        EditorFinding(
            title="Finding",
            source_url="https://example.com/finding",
            what_changed="Something changed",
            why_it_matters="It affects engineers",
            confidence=confidence,
            skeptic_objection="Evidence is preliminary",
        )


def test_triage_decision_requires_action_when_not_done() -> None:
    from models.schemas import TriageDecision

    with pytest.raises(ValidationError):
        TriageDecision(done=False)


def test_triage_decision_requires_summary_when_done() -> None:
    from models.schemas import TriageDecision

    with pytest.raises(ValidationError):
        TriageDecision(done=True)
