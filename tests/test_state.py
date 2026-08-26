import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.schemas import CandidateRecord, RunRecord, RunStatus, SourceItem
from storage.store import JsonStore


def make_candidate(candidate_id: str, terminal_status: str | None = None) -> CandidateRecord:
    return CandidateRecord(
        source=SourceItem(
            id=candidate_id,
            source_name="Official source",
            title=f"Announcement {candidate_id}",
            url=f"https://example.com/{candidate_id}",
            published_at=datetime(2026, 8, 25, tzinfo=UTC),
            content="Announcement body",
            content_hash=f"hash-{candidate_id}",
        ),
        terminal_status=terminal_status,
    )


def make_run(
    run_id: str,
    started_at: datetime,
    *,
    status: RunStatus = RunStatus.RUNNING,
    candidates: dict[str, CandidateRecord] | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        started_at=started_at,
        status=status,
        candidates=candidates or {},
    )


def make_store(tmp_path: Path) -> JsonStore:
    return JsonStore(tmp_path / "state", tmp_path / "runs")


def test_store_creates_directories_and_loads_missing_seen_items(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert (tmp_path / "state").is_dir()
    assert (tmp_path / "runs").is_dir()
    assert store.load_seen_items() == {}


def test_save_run_and_load_run_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    run = make_run(
        "run-1",
        datetime(2026, 8, 25, 10, tzinfo=UTC),
        status=RunStatus.SUCCESS,
        candidates={"candidate-1": make_candidate("candidate-1", "accept")},
    )

    path = store.save_run(run)

    assert path == tmp_path / "runs" / "run-1.json"
    assert store.load_run(path) == run


def test_latest_successful_run_ignores_failed_runs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    successful = make_run(
        "successful",
        datetime(2026, 8, 25, 10, tzinfo=UTC),
        status=RunStatus.SUCCESS,
    )
    failed = make_run(
        "failed",
        datetime(2026, 8, 25, 11, tzinfo=UTC),
        status=RunStatus.FAILED,
    )
    store.save_run(successful)
    store.save_run(failed)

    assert store.latest_successful_run() == successful


def test_latest_successful_run_returns_newest_success(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    older = make_run(
        "older",
        datetime(2026, 8, 25, 10, tzinfo=UTC),
        status=RunStatus.SUCCESS,
    )
    newest = make_run(
        "newest",
        datetime(2026, 8, 25, 11, tzinfo=UTC),
        status=RunStatus.SUCCESS,
    )
    store.save_run(newest)
    store.save_run(older)

    assert store.latest_successful_run() == newest


def test_latest_successful_run_returns_none_without_successful_run(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.save_run(
        make_run(
            "running",
            datetime(2026, 8, 25, 10, tzinfo=UTC),
            status=RunStatus.RUNNING,
        )
    )
    store.save_run(
        make_run(
            "failed",
            datetime(2026, 8, 25, 11, tzinfo=UTC),
            status=RunStatus.FAILED,
        )
    )

    assert store.latest_successful_run() is None


def test_mark_seen_preserves_existing_entries(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_seen("candidate-1", {"title": "First"})

    store.mark_seen("candidate-2", {"title": "Second"})

    assert store.load_seen_items() == {
        "candidate-1": {"title": "First"},
        "candidate-2": {"title": "Second"},
    }


def test_mark_seen_updates_existing_entry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_seen("candidate-1", {"title": "Old"})

    store.mark_seen("candidate-1", {"title": "New"})

    assert store.load_seen_items() == {"candidate-1": {"title": "New"}}


def test_mark_seen_many_merges_updates_without_losing_existing_entries(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.mark_seen_many({"candidate-1": {"title": "First"}})
    store.mark_seen_many({
        "candidate-2": {"title": "Second"},
        "candidate-1": {"title": "Updated"},
    })

    assert store.load_seen_items() == {
        "candidate-1": {"title": "Updated"},
        "candidate-2": {"title": "Second"},
    }


def test_mark_seen_many_no_op_for_empty_mapping(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_seen("candidate-1", {"title": "First"})

    store.mark_seen_many({})

    assert store.load_seen_items() == {"candidate-1": {"title": "First"}}


def test_atomic_writes_leave_valid_json(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_seen("candidate-1", {"title": "Announcement"})
    run_path = store.save_run(
        make_run("run-1", datetime(2026, 8, 25, 10, tzinfo=UTC))
    )

    with (tmp_path / "state" / "seen_items.json").open(encoding="utf-8") as file:
        assert json.load(file) == {"candidate-1": {"title": "Announcement"}}
    with run_path.open(encoding="utf-8") as file:
        assert json.load(file)["run_id"] == "run-1"
    assert not list(tmp_path.rglob("*.tmp"))


def test_iter_runs_loads_all_run_files_as_validated_records(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = make_run("first", datetime(2026, 8, 25, 10, tzinfo=UTC))
    second = make_run("second", datetime(2026, 8, 25, 11, tzinfo=UTC))
    store.save_run(first)
    store.save_run(second)

    assert {run.run_id for run in store.iter_runs()} == {"first", "second"}

    (tmp_path / "runs" / "invalid.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError):
        list(store.iter_runs())


def test_find_reusable_terminal_candidate_returns_completed_candidate(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    candidate = make_candidate("candidate-1", "accept")
    store.save_run(
        make_run(
            "failed",
            datetime(2026, 8, 25, 10, tzinfo=UTC),
            status=RunStatus.FAILED,
            candidates={"candidate-1": candidate},
        )
    )

    assert store.find_reusable_terminal_candidate("candidate-1") == candidate


def test_find_reusable_terminal_candidate_ignores_unfinished_candidate(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.save_run(
        make_run(
            "failed",
            datetime(2026, 8, 25, 10, tzinfo=UTC),
            status=RunStatus.FAILED,
            candidates={"candidate-1": make_candidate("candidate-1")},
        )
    )

    assert store.find_reusable_terminal_candidate("candidate-1") is None


def test_find_reusable_terminal_candidate_returns_newest_completed_copy(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    older = make_candidate("candidate-1", "reject")
    newest = make_candidate("candidate-1", "accept")
    store.save_run(
        make_run(
            "older",
            datetime(2026, 8, 25, 10, tzinfo=UTC),
            status=RunStatus.FAILED,
            candidates={"candidate-1": older},
        )
    )
    store.save_run(
        make_run(
            "newest",
            datetime(2026, 8, 25, 11, tzinfo=UTC),
            status=RunStatus.FAILED,
            candidates={"candidate-1": newest},
        )
    )

    assert store.find_reusable_terminal_candidate("candidate-1") == newest


def test_find_reusable_terminal_candidate_ignores_error_terminal_status(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.save_run(
        make_run(
            "failed",
            datetime(2026, 8, 25, 10, tzinfo=UTC),
            status=RunStatus.FAILED,
            candidates={"candidate-1": make_candidate("candidate-1", "error")},
        )
    )

    assert store.find_reusable_terminal_candidate("candidate-1") is None
