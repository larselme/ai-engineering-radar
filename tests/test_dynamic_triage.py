from datetime import UTC, datetime

import pytest

from models.schemas import SourceItem, TriageAction, TriageTool, TriageToolResult
from orchestration.triage import (
    TriageBudgetError,
    TriagePolicy,
    TriagePolicyError,
    TriageTimeoutError,
    run_dynamic_triage,
)


NOW = datetime(2026, 8, 25, tzinfo=UTC)


def make_item(candidate_id: str, title: str = "Signal") -> SourceItem:
    return SourceItem(
        id=candidate_id,
        source_name="Official",
        title=title,
        url=f"https://example.com/{candidate_id}",
        published_at=NOW,
        content="Vendor announces a preview release with limited production detail.",
        content_hash=f"hash-{candidate_id}",
    )


def policy(**overrides) -> TriagePolicy:
    defaults = {
        "allowed_tools": set(TriageTool),
        "max_steps": 3,
        "timeout_seconds": 10,
        "max_tokens": 100,
        "max_cost": 1,
        "allowed_network_scopes": {"none", "official_sources"},
    }
    defaults.update(overrides)
    return TriagePolicy(**defaults)


def test_dynamic_triage_varies_tool_sequence_by_candidate_input() -> None:
    def planner(item, trace, _notes):
        if not trace:
            if "risk" in item.title.lower():
                return {
                    "done": False,
                    "action": {
                        "tool": "risk_check",
                        "reason": "preview reliability uncertainty",
                        "network_scope": "none",
                    },
                }
            return {
                "done": False,
                "action": {
                    "tool": "summarize",
                    "reason": "establish baseline",
                    "network_scope": "none",
                },
            }
        return {"done": True, "final_summary": "triage complete"}

    def tool_runner(_item, action):
        return TriageToolResult(observation=f"ran {action.tool.value}", tokens_used=2)

    risk = run_dynamic_triage(
        make_item("risk", "High risk preview"),
        policy(),
        planner,
        tool_runner,
    )
    baseline = run_dynamic_triage(
        make_item("base", "Routine release"),
        policy(),
        planner,
        tool_runner,
    )

    assert [step.tool for step in risk.trace] == [TriageTool.RISK_CHECK]
    assert [step.tool for step in baseline.trace] == [TriageTool.SUMMARIZE]


def test_dynamic_triage_policy_blocks_disallowed_tool() -> None:
    def planner(_item, _trace, _notes):
        return {
            "done": False,
            "action": {
                "tool": "compare_sources",
                "reason": "cross-check sources",
                "network_scope": "official_sources",
            },
        }

    with pytest.raises(TriagePolicyError, match="tool not allowed"):
        run_dynamic_triage(
            make_item("blocked"),
            policy(allowed_tools={TriageTool.SUMMARIZE}),
            planner,
            lambda item, action: TriageToolResult(observation="ok"),
        )


def test_dynamic_triage_enforces_budget_limits_safely() -> None:
    def planner(_item, trace, _notes):
        if trace:
            return {"done": True, "final_summary": "done"}
        return {
            "done": False,
            "action": TriageAction(
                tool=TriageTool.SUMMARIZE,
                reason="capture baseline",
                network_scope="none",
            ),
        }

    def expensive_tool(_item, _action):
        return TriageToolResult(observation="large response", tokens_used=15)

    with pytest.raises(TriageBudgetError, match="token budget") as exc:
        run_dynamic_triage(
            make_item("budget"),
            policy(max_tokens=10),
            planner,
            expensive_tool,
        )

    assert len(exc.value.trace) == 1


def test_dynamic_triage_timeout_terminates_safely() -> None:
    monotonic_values = iter([0.0, 0.0, 0.0, 2.0])

    def monotonic_provider():
        return next(monotonic_values)

    def planner(_item, trace, _notes):
        if trace:
            return {"done": True, "final_summary": "done"}
        return {
            "done": False,
            "action": {
                "tool": "summarize",
                "reason": "first pass",
                "network_scope": "none",
            },
        }

    with pytest.raises(TriageTimeoutError, match="timeout budget") as exc:
        run_dynamic_triage(
            make_item("timeout"),
            policy(timeout_seconds=1),
            planner,
            lambda *_args: TriageToolResult(observation="ok", tokens_used=1),
            monotonic_provider=monotonic_provider,
        )

    assert len(exc.value.trace) == 1

def test_dynamic_triage_policy_blocks_forbidden_network_scope() -> None:
    def planner(_item, _trace, _notes):
        return {
            "done": False,
            "action": {
                "tool": "fact_check",
                "reason": "verify claim",
                "network_scope": "internet",
            },
        }

    with pytest.raises(TriagePolicyError, match="network scope"):
        run_dynamic_triage(
            make_item("scope"),
            policy(allowed_network_scopes={"none", "official_sources"}),
            planner,
            lambda *_args: TriageToolResult(observation="ok"),
        )
