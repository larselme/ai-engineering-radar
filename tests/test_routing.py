from datetime import UTC, datetime

import pytest

from models.schemas import (
    AnalysisResult,
    CandidateTerminalStatus,
    CritiqueResult,
    JudgeDecision,
    SourceItem,
)
from orchestration.graph import GraphInvariantError, process_candidate


def item() -> SourceItem:
    return SourceItem(
        id="item-1",
        source_name="Official source",
        title="New release",
        url="https://example.com/release",
        published_at=datetime(2026, 8, 25, tzinfo=UTC),
        content="The release adds a new capability.",
        content_hash="hash",
    )


def analysis(number: int = 1) -> AnalysisResult:
    return AnalysisResult(
        summary=f"Summary {number}",
        what_changed="A capability changed.",
        engineering_impact="Teams can use it.",
        classification="emerging_signal",
        confidence=0.8,
    )


def critique(number: int = 1) -> CritiqueResult:
    return CritiqueResult(
        objections=[f"Objection {number}"],
        unsupported_claims=[],
        hype_risk="low",
        alternative_interpretation="This may be incremental.",
    )


def decision(status: str, feedback: str = "") -> JudgeDecision:
    return JudgeDecision(
        status=status,
        reason=f"Decision: {status}",
        feedback=feedback,
        confidence=0.9,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("accept", CandidateTerminalStatus.ACCEPT),
        ("watchlist", CandidateTerminalStatus.WATCHLIST),
        ("reject", CandidateTerminalStatus.REJECT),
    ],
)
def test_terminal_paths(status, expected) -> None:
    calls = []

    def analyst(item, previous_analysis, judge_feedback):
        calls.append("analyst")
        return analysis()

    def skeptic(item, current_analysis):
        calls.append("skeptic")
        return critique()

    def judge(item, current_analysis, current_critique, allow_revision):
        calls.append(("judge", allow_revision))
        return decision(status)

    record = process_candidate(item(), analyst, skeptic, judge)

    assert calls == ["analyst", "skeptic", ("judge", True)]
    assert record.terminal_status is expected
    assert record.revision_count == 0


def test_one_revision_then_accept_passes_history_to_analyst() -> None:
    calls = []
    analyses = [analysis(1), analysis(2)]
    critiques = [critique(1), critique(2)]
    seen_revision_args = []

    def analyst(item, previous_analysis, judge_feedback):
        calls.append("analyst")
        seen_revision_args.append((previous_analysis, judge_feedback))
        return analyses.pop(0)

    def skeptic(item, current_analysis):
        calls.append("skeptic")
        return critiques.pop(0)

    decisions = iter([decision("revise", "Clarify the evidence."), decision("accept")])

    def judge(item, current_analysis, current_critique, allow_revision):
        calls.append(("judge", allow_revision))
        return next(decisions)

    record = process_candidate(item(), analyst, skeptic, judge)

    assert calls == [
        "analyst",
        "skeptic",
        ("judge", True),
        "analyst",
        "skeptic",
        ("judge", True),
    ]
    assert seen_revision_args == [(None, None), (analysis(1), "Clarify the evidence.")]
    assert record.revision_count == 1
    assert len(record.analyses) == len(record.critiques) == len(record.decisions) == 2


def test_two_revisions_then_watchlist_reaches_revision_limit() -> None:
    calls = []
    decisions = iter(
        [decision("revise", "First"), decision("revise", "Second"), decision("watchlist")]
    )

    def analyst(item, previous_analysis, judge_feedback):
        calls.append("analyst")
        return analysis(len([call for call in calls if call == "analyst"]))

    def skeptic(item, current_analysis):
        calls.append("skeptic")
        return critique()

    def judge(item, current_analysis, current_critique, allow_revision):
        calls.append(("judge", allow_revision))
        return next(decisions)

    record = process_candidate(item(), analyst, skeptic, judge)

    assert calls.count("analyst") == 3
    assert calls.count("skeptic") == 3
    assert [call for call in calls if isinstance(call, tuple)] == [
        ("judge", True),
        ("judge", True),
        ("judge", False),
    ]
    assert record.revision_count == 2
    assert record.terminal_status is CandidateTerminalStatus.WATCHLIST


def test_illegal_revision_after_limit_raises_invariant_error() -> None:
    def analyst(item, previous_analysis, judge_feedback):
        return analysis()

    def skeptic(item, current_analysis):
        return critique()

    def judge(item, current_analysis, current_critique, allow_revision):
        return decision("revise", "Still revise.")

    with pytest.raises(GraphInvariantError):
        process_candidate(item(), analyst, skeptic, judge, max_revisions=0)


def test_negative_max_revisions_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_revisions"):
        process_candidate(item(), lambda *_: analysis(), lambda *_: critique(), lambda *_: decision("accept"), -1)


def test_analyst_exception_returns_error() -> None:
    def analyst(item, previous_analysis, judge_feedback):
        raise RuntimeError("boom")

    record = process_candidate(item(), analyst, lambda *_: critique(), lambda *_: decision("accept"))

    assert record.terminal_status is CandidateTerminalStatus.ERROR
    assert record.error == "RuntimeError: boom"


def test_skeptic_exception_preserves_analysis() -> None:
    def analyst(item, previous_analysis, judge_feedback):
        return analysis()

    def skeptic(item, current_analysis):
        raise RuntimeError("skeptic boom")

    record = process_candidate(item(), analyst, skeptic, lambda *_: decision("accept"))

    assert record.terminal_status is CandidateTerminalStatus.ERROR
    assert record.analyses == [analysis()]
    assert record.critiques == []


def test_judge_exception_preserves_analysis_and_critique() -> None:
    def analyst(item, previous_analysis, judge_feedback):
        return analysis()

    def skeptic(item, current_analysis):
        return critique()

    def judge(item, current_analysis, current_critique, allow_revision):
        raise RuntimeError("judge boom")

    record = process_candidate(item(), analyst, skeptic, judge)

    assert record.terminal_status is CandidateTerminalStatus.ERROR
    assert record.analyses == [analysis()]
    assert record.critiques == [critique()]
    assert record.decisions == []


def test_revision_analyst_exception_preserves_first_pass() -> None:
    calls = 0

    def analyst(item, previous_analysis, judge_feedback):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("revision boom")
        return analysis()

    record = process_candidate(
        item(),
        analyst,
        lambda *_: critique(),
        lambda *_: decision("revise", "Improve it."),
    )

    assert record.terminal_status is CandidateTerminalStatus.ERROR
    assert record.revision_count == 1
    assert record.analyses == [analysis()]
    assert record.critiques == [critique()]
    assert record.decisions == [decision("revise", "Improve it.")]
