from datetime import UTC, datetime
import pytest
from pydantic import ValidationError

from agents.analyst import run_analyst
from agents.client import StructuredCopilotClient
from agents.editor import run_editor
from agents.judge import run_judge
from agents.skeptic import run_skeptic
from models.schemas import (
    AnalysisResult,
    CandidateRecord,
    CritiqueResult,
    EditorReport,
    JudgeDecision,
    SourceItem,
    TriageTraceStep,
    TriageTool,
)


def source_item() -> SourceItem:
    return SourceItem(
        id="item-1",
        source_name="Official source",
        title="New release",
        url="https://example.com/release",
        published_at=datetime(2026, 8, 25, tzinfo=UTC),
        content="The release adds a new capability.",
        content_hash="hash",
    )


def analysis() -> AnalysisResult:
    return AnalysisResult(
        summary="A new capability was released.",
        what_changed="The product gained a capability.",
        engineering_impact="Teams can use it.",
        classification="emerging_signal",
        confidence=0.8,
    )


def critique() -> CritiqueResult:
    return CritiqueResult(
        objections=["Evidence is limited."],
        unsupported_claims=[],
        hype_risk="low",
        alternative_interpretation="This may be incremental.",
    )


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, str, type]] = []

    def parse(self, model, prompt, result_type):
        self.calls.append((model, prompt, result_type))
        return self.result


def test_analyst_asks_for_analysis_result() -> None:
    client = FakeClient(analysis())

    result = run_analyst(client, "analyst-model", source_item())

    assert result == analysis()
    assert client.calls[0][2] is AnalysisResult


def test_analyst_revision_prompt_includes_previous_analysis_and_feedback() -> None:
    client = FakeClient(analysis())

    run_analyst(
        client,
        "analyst-model",
        source_item(),
        previous_analysis=analysis(),
        judge_feedback="Clarify the availability status.",
    )

    prompt = client.calls[0][1]
    assert "Clarify the availability status." in prompt
    assert analysis().summary in prompt


def test_triage_evidence_is_included_in_agent_prompts() -> None:
    trace = [
        TriageTraceStep(
            step_number=1,
            tool=TriageTool.SUMMARIZE,
            reason="baseline",
            network_scope="none",
            started_at=datetime(2026, 8, 25, tzinfo=UTC),
            completed_at=datetime(2026, 8, 25, tzinfo=UTC),
            observation="summary",
            tokens_used=1,
            estimated_cost=0,
        )
    ]
    triage_summary = "triage summary"

    analyst_client = FakeClient(analysis())
    run_analyst(
        analyst_client,
        "analyst-model",
        source_item(),
        triage_summary=triage_summary,
        triage_trace=trace,
    )
    assert "TRIAGE EVIDENCE" in analyst_client.calls[0][1]
    assert triage_summary in analyst_client.calls[0][1]

    skeptic_client = FakeClient(critique())
    run_skeptic(
        skeptic_client,
        "skeptic-model",
        source_item(),
        analysis(),
        triage_summary=triage_summary,
        triage_trace=trace,
    )
    assert "TRIAGE EVIDENCE" in skeptic_client.calls[0][1]
    assert triage_summary in skeptic_client.calls[0][1]

    judge_client = FakeClient(
        JudgeDecision(status="accept", reason="Strong evidence.", confidence=0.9)
    )
    run_judge(
        judge_client,
        "judge-model",
        source_item(),
        analysis(),
        critique(),
        allow_revision=True,
        triage_summary=triage_summary,
        triage_trace=trace,
    )
    assert "TRIAGE EVIDENCE" in judge_client.calls[0][1]
    assert triage_summary in judge_client.calls[0][1]


def test_skeptic_asks_for_critique_and_includes_source_and_analysis() -> None:
    client = FakeClient(critique())

    run_skeptic(client, "skeptic-model", source_item(), analysis())

    prompt = client.calls[0][1]
    assert client.calls[0][2] is CritiqueResult
    assert source_item().content in prompt
    assert analysis().what_changed in prompt


def test_judge_asks_for_decision_and_states_revision_allowed() -> None:
    decision = JudgeDecision(
        status="accept", reason="Strong evidence.", confidence=0.9
    )
    client = FakeClient(decision)

    run_judge(
        client,
        "judge-model",
        source_item(),
        analysis(),
        critique(),
        allow_revision=True,
    )

    assert client.calls[0][2] is JudgeDecision
    assert "revise is currently allowed" in client.calls[0][1].lower()


def test_judge_forbids_revision_when_not_allowed() -> None:
    client = FakeClient(
        JudgeDecision(status="reject", reason="Weak evidence.", confidence=0.8)
    )

    run_judge(
        client,
        "judge-model",
        source_item(),
        analysis(),
        critique(),
        allow_revision=False,
    )

    assert "must not choose revise" in client.calls[0][1].lower()


def test_editor_asks_for_report_and_receives_only_passed_candidates() -> None:
    accepted = [CandidateRecord(source=source_item(), analyses=[analysis()])]
    watchlist = [
        CandidateRecord(
            source=source_item().model_copy(update={"id": "watch-1"}),
            analyses=[analysis()],
        )
    ]
    client = FakeClient(EditorReport(top_findings=[], watchlist=[]))

    run_editor(client, "editor-model", accepted, watchlist)

    prompt = client.calls[0][1]
    assert client.calls[0][2] is EditorReport
    assert "watch-1" in prompt
    assert "item-1" in prompt
    assert "REJECTED CANDIDATES:" not in prompt


def test_client_keeps_copilot_auth_configuration() -> None:
    client = StructuredCopilotClient("test-token", use_logged_in_user=False)

    assert client._github_token == "test-token"
    assert client._use_logged_in_user is False


def test_transient_failures_are_retried_at_most_three_attempts(monkeypatch) -> None:
    calls = 0

    async def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary")

    client = StructuredCopilotClient("test-token")
    monkeypatch.setattr(client, "_send_prompt", fail)
    monkeypatch.setattr("agents.client.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="temporary"):
        client.parse("model", "prompt", AnalysisResult)
    assert calls == 3


def test_missing_parsed_output_is_retried_once_then_raised(
    monkeypatch,
) -> None:
    calls = 0

    async def respond(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "not json"

    client = StructuredCopilotClient("test-token")
    monkeypatch.setattr(client, "_send_prompt", respond)
    monkeypatch.setattr("agents.client.time.sleep", lambda _: None)

    with pytest.raises(ValueError, match="valid JSON"):
        client.parse("model", "prompt", AnalysisResult)
    assert calls == 2


def test_invalid_parsed_output_is_retried_once_then_raised(
    monkeypatch,
) -> None:
    calls = 0

    async def respond(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return '{"confidence": "not-a-number"}'

    client = StructuredCopilotClient("test-token")
    monkeypatch.setattr(client, "_send_prompt", respond)
    monkeypatch.setattr("agents.client.time.sleep", lambda _: None)

    with pytest.raises(ValidationError):
        client.parse("model", "prompt", AnalysisResult)
    assert calls == 2
