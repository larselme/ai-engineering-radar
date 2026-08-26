from datetime import UTC, datetime, timedelta

import pytest

import main
from models.schemas import (
    AnalysisResult,
    AnalysisClassification,
    CandidateRecord,
    CandidateTerminalStatus,
    CritiqueResult,
    EditorReport,
    HypeRisk,
    JudgeDecision,
    JudgeStatus,
    RunRecord,
    RunStatus,
    SourceConfig,
    SourceItem,
    TriageOutcome,
    TriageTraceStep,
    TriageTool,
)
from orchestration.graph import GraphInvariantError
from orchestration.triage import TriageBudgetError
from storage.store import JsonStore


NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def settings():
    from config import Settings

    return Settings(copilot_github_token="test-token", use_logged_in_copilot=False)


def item(candidate_id: str) -> SourceItem:
    return SourceItem(
        id=candidate_id,
        source_name="Official",
        title=candidate_id,
        url=f"https://example.com/{candidate_id}",
        published_at=NOW,
        content="content",
        content_hash=f"hash-{candidate_id}",
    )


def candidate(candidate_id: str, status: CandidateTerminalStatus) -> CandidateRecord:
    return CandidateRecord(source=item(candidate_id), terminal_status=status)


def fake_report(*_args):
    return EditorReport(top_findings=[], watchlist=[])


def setup_run(monkeypatch, items, process=None, editor=fake_report):
    monkeypatch.setattr(
        main,
        "_load_sources",
        lambda: [SourceConfig(name="Official", kind="rss", url="https://example.com")],
    )
    monkeypatch.setattr(main, "collect_sources", lambda sources, since: (items, {}))
    monkeypatch.setattr(
        main,
        "StructuredCopilotClient",
        lambda token, use_logged_in_user=True: object(),
    )
    monkeypatch.setattr(
        main,
        "run_dynamic_triage",
        lambda *args, **kwargs: TriageOutcome(summary="triaged", trace=[]),
    )
    monkeypatch.setattr(main, "run_editor", editor)
    if process is not None:
        monkeypatch.setattr(main, "process_candidate", process)


def test_first_run_uses_seven_day_window_and_validates_sources(monkeypatch):
    seen = {}

    def collect(sources, since):
        seen["sources"] = sources
        seen["since"] = since
        return [], {}

    monkeypatch.setattr(
        main,
        "_load_sources",
        lambda: [SourceConfig(name="Official", kind="rss", url="https://example.com")],
    )
    monkeypatch.setattr(main, "collect_sources", collect)
    monkeypatch.setattr(
        main,
        "StructuredCopilotClient",
        lambda token, use_logged_in_user=True: object(),
    )
    monkeypatch.setattr(
        main,
        "run_dynamic_triage",
        lambda *args, **kwargs: TriageOutcome(summary="triaged", trace=[]),
    )
    monkeypatch.setattr(main, "run_editor", fake_report)

    run = main.run_radar(settings(), NOW)

    assert seen["since"] == NOW - timedelta(days=7)
    assert all(isinstance(source, SourceConfig) for source in seen["sources"])
    assert run.status is RunStatus.SUCCESS


def test_source_errors_are_retained_without_failing_run(monkeypatch):
    setup_run(monkeypatch, [])
    monkeypatch.setattr(
        main,
        "collect_sources",
        lambda sources, since: ([], {"Official": "OSError: offline"}),
    )

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.SUCCESS
    assert run.source_errors == {"Official": "OSError: offline"}


@pytest.mark.parametrize("ended_at", [NOW - timedelta(hours=1), None])
def test_previous_success_is_audit_metadata_not_collection_watermark(monkeypatch, ended_at):
    previous = RunRecord(
        run_id="previous",
        started_at=NOW - timedelta(days=2),
        ended_at=ended_at,
        status=RunStatus.SUCCESS,
    )
    captured = {}
    setup_run(monkeypatch, [])
    store = JsonStore(main.STATE_DIR, main.RUNS_DIR)
    store.save_run(previous)
    monkeypatch.setattr(
        main,
        "collect_sources",
        lambda sources, since: (captured.setdefault("since", since) and ([], {}) or ([], {})),
    )

    run = main.run_radar(settings(), NOW)

    expected_previous = ended_at or previous.started_at
    assert captured["since"] == NOW - timedelta(days=7)
    assert run.previous_successful_run == expected_previous
    assert run.metadata["collection_since"] == (NOW - timedelta(days=7)).isoformat()


def test_successful_run_uses_single_batch_seen_update(monkeypatch):
    items = [item("accept"), item("reject")]
    statuses = iter(
        [
            CandidateTerminalStatus.ACCEPT,
            CandidateTerminalStatus.REJECT,
        ]
    )
    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: candidate(current.id, next(statuses)),
    )

    seen_updates = {}
    monkeypatch.setattr(
        JsonStore,
        "mark_seen_many",
        lambda self, updates: seen_updates.setdefault("updates", updates.copy()) or None,
    )

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.SUCCESS
    assert set(seen_updates["updates"]) == {"accept", "reject"}
    assert all(payload["run_id"] == run.run_id for payload in seen_updates["updates"].values())


def test_naive_now_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        main.run_radar(settings(), datetime(2026, 8, 25, 12))


def test_seen_and_reusable_candidates_avoid_processing(monkeypatch):
    items = [item("seen"), item("reused"), item("new")]
    calls = []
    reusable = candidate("reused", CandidateTerminalStatus.ACCEPT)
    setup_run(
        monkeypatch,
        items,
        lambda *args, **kwargs: calls.append(args)
        or candidate(args[0].id, CandidateTerminalStatus.WATCHLIST),
    )
    store = JsonStore(main.STATE_DIR, main.RUNS_DIR)
    store.mark_seen("seen", {"old": True})
    store.save_run(
        RunRecord(
            run_id="failed",
            started_at=NOW - timedelta(hours=1),
            status=RunStatus.FAILED,
            candidates={"reused": reusable},
        )
    )
    run = main.run_radar(settings(), NOW)

    assert [args[0].id for args in calls] == ["new"]
    assert set(run.candidates) == {"reused", "new"}
    assert run.accepted_ids == ["reused"]


def test_triage_failure_becomes_candidate_error_and_later_candidates_continue(monkeypatch):
    items = [item("triage-bad"), item("triage-good")]

    def triage(current, *_args, **_kwargs):
        if current.id == "triage-bad":
            raise TriageBudgetError("triage token budget exceeded")
        return TriageOutcome(
            summary="triaged",
            trace=[
                TriageTraceStep(
                    step_number=1,
                    tool=TriageTool.SUMMARIZE,
                    reason="initial overview",
                    network_scope="none",
                    started_at=NOW,
                    completed_at=NOW,
                    observation="ok",
                    tokens_used=1,
                    estimated_cost=0,
                )
            ],
        )

    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: candidate(current.id, CandidateTerminalStatus.ACCEPT),
    )
    monkeypatch.setattr(main, "run_dynamic_triage", triage)

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.SUCCESS
    assert run.candidates["triage-bad"].terminal_status is CandidateTerminalStatus.ERROR
    assert run.candidates["triage-good"].terminal_status is CandidateTerminalStatus.ACCEPT
    assert run.error_ids == ["triage-bad"]
    assert run.accepted_ids == ["triage-good"]


@pytest.mark.parametrize(
    ("status", "bucket"),
    [
        (CandidateTerminalStatus.ACCEPT, "accepted_ids"),
        (CandidateTerminalStatus.WATCHLIST, "watchlist_ids"),
        (CandidateTerminalStatus.REJECT, "rejected_ids"),
        (CandidateTerminalStatus.ERROR, "error_ids"),
    ],
)
def test_candidates_are_put_in_exactly_one_bucket(monkeypatch, status, bucket):
    items = [item("one"), item("two")]
    statuses = iter([status, CandidateTerminalStatus.ACCEPT])
    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: candidate(current.id, next(statuses)),
    )

    run = main.run_radar(settings(), NOW)

    if status is CandidateTerminalStatus.ACCEPT:
        assert "one" in run.accepted_ids
    else:
        assert getattr(run, bucket) == ["one"]
    for candidate_id in ("one", "two"):
        memberships = sum(
            candidate_id in getattr(run, name)
            for name in ("accepted_ids", "watchlist_ids", "rejected_ids", "error_ids")
        )
        assert memberships == 1


def test_editor_gets_only_accepted_and_watchlist_and_success_marks_all_seen(monkeypatch):
    items = [item("accept"), item("watch"), item("reject"), item("error")]
    statuses = iter(
        [
            CandidateTerminalStatus.ACCEPT,
            CandidateTerminalStatus.WATCHLIST,
            CandidateTerminalStatus.REJECT,
            CandidateTerminalStatus.ERROR,
        ]
    )
    received = {}
    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: candidate(current.id, next(statuses)),
        lambda client, model, accepted, watchlist: (
            received.update(
                accepted=[c.source.id for c in accepted],
                watchlist=[c.source.id for c in watchlist],
            )
            or fake_report()
        ),
    )

    run = main.run_radar(settings(), NOW)
    seen = JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items()

    assert received == {"accepted": ["accept"], "watchlist": ["watch"]}
    assert run.status is RunStatus.SUCCESS
    assert set(seen) == {"accept", "watch", "reject"}
    assert "error" not in seen
    assert all(payload["run_id"] == run.run_id for payload in seen.values())


def test_triage_evidence_is_passed_to_analysis_graph_agents(monkeypatch):
    captured: dict[str, object] = {}
    triage_step = TriageTraceStep(
        step_number=1,
        tool=TriageTool.SUMMARIZE,
        reason="baseline",
        network_scope="none",
        started_at=NOW,
        completed_at=NOW,
        observation="summary",
        tokens_used=1,
        estimated_cost=0,
    )
    triage = TriageOutcome(summary="triage summary", trace=[triage_step])

    monkeypatch.setattr(
        main,
        "_load_sources",
        lambda: [SourceConfig(name="Official", kind="rss", url="https://example.com")],
    )
    monkeypatch.setattr(main, "collect_sources", lambda sources, since: ([item("one")], {}))
    monkeypatch.setattr(
        main,
        "StructuredCopilotClient",
        lambda token, use_logged_in_user=True: object(),
    )
    monkeypatch.setattr(main, "run_dynamic_triage", lambda *args, **kwargs: triage)
    monkeypatch.setattr(main, "run_editor", fake_report)

    def analyst(
        _client,
        _model,
        _item,
        _previous_analysis=None,
        _judge_feedback=None,
        triage_summary=None,
        triage_trace=None,
    ):
        captured["analyst_summary"] = triage_summary
        captured["analyst_trace"] = triage_trace
        return AnalysisResult(
            summary="Summary",
            what_changed="Changed",
            engineering_impact="Impact",
            classification=AnalysisClassification.EMERGING_SIGNAL,
            confidence=0.8,
        )

    def skeptic(
        _client,
        _model,
        _item,
        _analysis,
        triage_summary=None,
        triage_trace=None,
    ):
        captured["skeptic_summary"] = triage_summary
        captured["skeptic_trace"] = triage_trace
        return CritiqueResult(
            objections=["Need better evidence"],
            unsupported_claims=[],
            hype_risk=HypeRisk.LOW,
            alternative_interpretation="Could be incremental",
        )

    def judge(
        _client,
        _model,
        _item,
        _analysis,
        _critique,
        _allow_revision,
        triage_summary=None,
        triage_trace=None,
    ):
        captured["judge_summary"] = triage_summary
        captured["judge_trace"] = triage_trace
        return JudgeDecision(
            status=JudgeStatus.ACCEPT,
            reason="Enough evidence",
            confidence=0.9,
        )

    monkeypatch.setattr(main, "run_analyst", analyst)
    monkeypatch.setattr(main, "run_skeptic", skeptic)
    monkeypatch.setattr(main, "run_judge", judge)

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.SUCCESS
    assert captured["analyst_summary"] == "triage summary"
    assert captured["skeptic_summary"] == "triage summary"
    assert captured["judge_summary"] == "triage summary"
    assert captured["analyst_trace"] == [triage_step]
    assert captured["skeptic_trace"] == [triage_step]
    assert captured["judge_trace"] == [triage_step]


def test_error_candidates_are_retried_not_seen_or_reused(monkeypatch):
    monkeypatch.setattr(
        main,
        "_load_sources",
        lambda: [SourceConfig(name="Official", kind="rss", url="https://example.com")],
    )
    monkeypatch.setattr(main, "collect_sources", lambda sources, since: ([item("retry-me")], {}))
    monkeypatch.setattr(
        main,
        "StructuredCopilotClient",
        lambda token, use_logged_in_user=True: object(),
    )
    monkeypatch.setattr(
        main,
        "run_dynamic_triage",
        lambda *args, **kwargs: TriageOutcome(summary="triaged", trace=[]),
    )
    monkeypatch.setattr(main, "run_editor", fake_report)

    calls = 0

    def process(_item, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        status = (
            CandidateTerminalStatus.ERROR
            if calls == 1
            else CandidateTerminalStatus.ACCEPT
        )
        return candidate("retry-me", status)

    monkeypatch.setattr(main, "process_candidate", process)

    first = main.run_radar(settings(), NOW)
    second = main.run_radar(settings(), NOW + timedelta(minutes=1))
    seen = JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items()

    assert first.status is RunStatus.SUCCESS
    assert first.candidates["retry-me"].terminal_status is CandidateTerminalStatus.ERROR
    assert second.status is RunStatus.SUCCESS
    assert second.candidates["retry-me"].terminal_status is CandidateTerminalStatus.ACCEPT
    assert calls == 2
    assert seen["retry-me"]["terminal_status"] == CandidateTerminalStatus.ACCEPT.value


@pytest.mark.parametrize("failure", [RuntimeError("editor"), ValueError("report")])
def test_editor_or_validation_failure_does_not_mark_seen(monkeypatch, failure):
    setup_run(monkeypatch, [item("candidate")])
    monkeypatch.setattr(
        main,
        "process_candidate",
        lambda current, *args, **kwargs: candidate(
            current.id, CandidateTerminalStatus.ACCEPT
        ),
    )
    if isinstance(failure, RuntimeError):
        monkeypatch.setattr(main, "run_editor", lambda *args: (_ for _ in ()).throw(failure))
    else:
        monkeypatch.setattr(
            main,
            "validate_editor_report",
            lambda *args: (_ for _ in ()).throw(failure),
        )

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.FAILED
    assert JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items() == {}


def test_publication_failure_does_not_mark_seen(monkeypatch):
    setup_run(monkeypatch, [item("candidate")])
    monkeypatch.setattr(
        main,
        "process_candidate",
        lambda current, *args, **kwargs: candidate(
            current.id, CandidateTerminalStatus.ACCEPT
        ),
    )
    monkeypatch.setattr(
        main,
        "publish_report",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.FAILED
    assert JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items() == {}


def test_graph_invariant_fails_run_and_stops_later_candidates(monkeypatch):
    items = [item("first"), item("later")]
    calls = []
    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: (
            calls.append(current.id) or (_ for _ in ()).throw(GraphInvariantError("bad"))
        ),
    )
    monkeypatch.setattr(main, "run_editor", lambda *args: pytest.fail("editor called"))

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.FAILED
    assert "GraphInvariantError: bad" == run.metadata["fatal_error"]
    assert calls == ["first"]
    assert JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items() == {}


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(main, "load_settings", settings)
    monkeypatch.setattr(
        main,
        "run_radar",
        lambda _: RunRecord(run_id="ok", started_at=NOW, status=RunStatus.SUCCESS),
    )
    assert main.main() == 0
    monkeypatch.setattr(
        main,
        "run_radar",
        lambda _: RunRecord(run_id="bad", started_at=NOW, status=RunStatus.FAILED),
    )
    assert main.main() == 1


def test_main_returns_one_for_unexpected_run_exception(monkeypatch):
    monkeypatch.setattr(main, "load_settings", settings)
    monkeypatch.setattr(
        main,
        "run_radar",
        lambda _: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    assert main.main() == 1
