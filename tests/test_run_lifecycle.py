from datetime import UTC, datetime, timedelta
import pytest

import main
from models.schemas import (
    AnalysisResult,
    CandidateRecord,
    CandidateTerminalStatus,
    CritiqueResult,
    EditorReport,
    JudgeDecision,
    RunRecord,
    RunStatus,
    SourceConfig,
    SourceItem,
)
from orchestration.graph import GraphInvariantError
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
    monkeypatch.setattr(main, "run_editor", editor)
    if process is not None:
        monkeypatch.setattr(main, "process_candidate", process)


def test_first_run_uses_seven_day_window_and_validates_sources(monkeypatch):
    seen = {}

    def collect(sources, since):
        seen["sources"] = sources
        seen["since"] = since
        return [], {}

    monkeypatch.setattr(main, "_load_sources", lambda: [
        SourceConfig(name="Official", kind="rss", url="https://example.com")
    ])
    monkeypatch.setattr(main, "collect_sources", collect)
    monkeypatch.setattr(
        main,
        "StructuredCopilotClient",
        lambda token, use_logged_in_user=True: object(),
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
    monkeypatch.setattr(main, "collect_sources", lambda sources, since: (
        captured.setdefault("since", since) and ([], {}) or ([], {})
    ))

    run = main.run_radar(settings(), NOW)

    expected_previous = ended_at or previous.started_at
    assert captured["since"] == NOW - timedelta(days=7)
    assert run.previous_successful_run == expected_previous
    assert run.metadata["collection_since"] == (NOW - timedelta(days=7)).isoformat()


def test_successful_run_uses_single_batch_seen_update(monkeypatch):
    items = [item("accept"), item("reject")]
    statuses = iter([
        CandidateTerminalStatus.ACCEPT,
        CandidateTerminalStatus.REJECT,
    ])
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
    setup_run(monkeypatch, items, lambda *args, **kwargs: calls.append(args) or candidate(
        args[0].id, CandidateTerminalStatus.WATCHLIST
    ))
    store = JsonStore(main.STATE_DIR, main.RUNS_DIR)
    store.mark_seen("seen", {"old": True})
    store.save_run(RunRecord(
        run_id="failed",
        started_at=NOW - timedelta(hours=1),
        status=RunStatus.FAILED,
        candidates={"reused": reusable},
    ))
    run = main.run_radar(settings(), NOW)

    assert [args[0].id for args in calls] == ["new"]
    assert set(run.candidates) == {"reused", "new"}
    assert run.accepted_ids == ["reused"]


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
    statuses = iter([
        CandidateTerminalStatus.ACCEPT,
        CandidateTerminalStatus.WATCHLIST,
        CandidateTerminalStatus.REJECT,
        CandidateTerminalStatus.ERROR,
    ])
    received = {}
    setup_run(
        monkeypatch,
        items,
        lambda current, *args, **kwargs: candidate(current.id, next(statuses)),
        lambda client, model, accepted, watchlist: (
            received.update(accepted=[c.source.id for c in accepted],
                            watchlist=[c.source.id for c in watchlist])
            or fake_report()
        ),
    )

    run = main.run_radar(settings(), NOW)
    seen = JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items()

    assert received == {"accepted": ["accept"], "watchlist": ["watch"]}
    assert run.status is RunStatus.SUCCESS
    assert set(seen) == {"accept", "watch", "reject", "error"}
    assert all(payload["run_id"] == run.run_id for payload in seen.values())


@pytest.mark.parametrize("failure", [RuntimeError("editor"), ValueError("report")])
def test_editor_or_validation_failure_does_not_mark_seen(monkeypatch, failure):
    setup_run(monkeypatch, [item("candidate")])
    monkeypatch.setattr(
        main,
        "process_candidate",
        lambda current, *args, **kwargs: candidate(current.id, CandidateTerminalStatus.ACCEPT),
    )
    if isinstance(failure, RuntimeError):
        monkeypatch.setattr(main, "run_editor", lambda *args: (_ for _ in ()).throw(failure))
    else:
        monkeypatch.setattr(main, "validate_editor_report", lambda *args: (_ for _ in ()).throw(failure))

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
    setup_run(monkeypatch, items,     lambda current, *args, **kwargs: (
        calls.append(current.id) or (_ for _ in ()).throw(GraphInvariantError("bad"))
    ))
    monkeypatch.setattr(main, "run_editor", lambda *args: pytest.fail("editor called"))

    run = main.run_radar(settings(), NOW)

    assert run.status is RunStatus.FAILED
    assert "GraphInvariantError: bad" == run.metadata["fatal_error"]
    assert calls == ["first"]
    assert JsonStore(main.STATE_DIR, main.RUNS_DIR).load_seen_items() == {}


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(main, "load_settings", settings)
    monkeypatch.setattr(main, "run_radar", lambda _: RunRecord(
        run_id="ok", started_at=NOW, status=RunStatus.SUCCESS
    ))
    assert main.main() == 0
    monkeypatch.setattr(main, "run_radar", lambda _: RunRecord(
        run_id="bad", started_at=NOW, status=RunStatus.FAILED
    ))
    assert main.main() == 1


def test_main_returns_one_for_unexpected_run_exception(monkeypatch):
    monkeypatch.setattr(main, "load_settings", settings)
    monkeypatch.setattr(
        main, "run_radar", lambda _: (_ for _ in ()).throw(RuntimeError("unexpected"))
    )

    assert main.main() == 1
