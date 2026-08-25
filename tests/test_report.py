from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from models.schemas import (
    CandidateRecord,
    EditorFinding,
    EditorReport,
    RunRecord,
    SourceItem,
)
from reporting.markdown import (
    ReportValidationError,
    build_editor_inputs,
    publish_report,
    render_markdown,
    validate_editor_report,
)


def candidate(candidate_id: str, status: str) -> CandidateRecord:
    return CandidateRecord(
        source=SourceItem(
            id=candidate_id,
            source_name="Source",
            title=f"Source {candidate_id}",
            url=f"https://example.com/{candidate_id}",
            published_at=datetime(2026, 8, 25, tzinfo=UTC),
            content="Content",
            content_hash=f"hash-{candidate_id}",
        ),
        terminal_status=status,
    )


@pytest.fixture
def run() -> RunRecord:
    return RunRecord(
        run_id="run-1",
        started_at=datetime(2026, 8, 25, tzinfo=UTC),
        candidates={
            "accepted-1": candidate("accepted-1", "accept"),
            "accepted-2": candidate("accepted-2", "accept"),
            "watchlist-1": candidate("watchlist-1", "watchlist"),
            "rejected-1": candidate("rejected-1", "reject"),
            "error-1": candidate("error-1", "error"),
        },
        accepted_ids=["accepted-2", "accepted-1"],
        watchlist_ids=["watchlist-1"],
        rejected_ids=["rejected-1"],
        error_ids=["error-1"],
    )


def finding(url: str, confidence: float = 0.88, title: str = "Finding") -> EditorFinding:
    return EditorFinding(
        title=title,
        source_url=url,
        what_changed="A meaningful change",
        why_it_matters="It matters to engineers",
        confidence=confidence,
        skeptic_objection="The evidence may be early",
    )


def test_build_editor_inputs_uses_ordered_status_buckets(run: RunRecord) -> None:
    accepted, watchlist = build_editor_inputs(run)

    assert [item.source.id for item in accepted] == ["accepted-2", "accepted-1"]
    assert [item.source.id for item in watchlist] == ["watchlist-1"]


def test_build_editor_inputs_excludes_rejected_and_error_candidates(
    run: RunRecord,
) -> None:
    run.accepted_ids += ["rejected-1", "error-1"]
    run.watchlist_ids += ["rejected-1", "error-1"]

    accepted, watchlist = build_editor_inputs(run)

    assert {item.source.id for item in accepted + watchlist} == {
        "accepted-1",
        "accepted-2",
        "watchlist-1",
    }


@pytest.mark.parametrize(
    ("section", "url"),
    [
        ("top_findings", "https://example.com/watchlist-1"),
        ("watchlist", "https://example.com/accepted-1"),
        ("top_findings", "https://example.com/rejected-1"),
        ("watchlist", "https://example.com/error-1"),
        ("top_findings", "https://example.com/unknown"),
    ],
)
def test_validate_editor_report_rejects_invalid_provenance(
    run: RunRecord, section: str, url: str
) -> None:
    report = EditorReport(
        top_findings=[finding(url)] if section == "top_findings" else [],
        watchlist=[finding(url)] if section == "watchlist" else [],
    )

    with pytest.raises(ReportValidationError):
        validate_editor_report(report, run)


def test_validate_editor_report_accepts_matching_buckets(run: RunRecord) -> None:
    validate_editor_report(
        EditorReport(
            top_findings=[finding("https://example.com/accepted-1")],
            watchlist=[finding("https://example.com/watchlist-1")],
        ),
        run,
    )


@pytest.mark.parametrize("section", ["top_findings", "watchlist"])
def test_validate_editor_report_rejects_duplicates(run: RunRecord, section: str) -> None:
    url = (
        "https://example.com/accepted-1"
        if section == "top_findings"
        else "https://example.com/watchlist-1"
    )
    report = EditorReport(
        top_findings=[finding(url), finding(url)] if section == "top_findings" else [],
        watchlist=[finding(url), finding(url)] if section == "watchlist" else [],
    )

    with pytest.raises(ReportValidationError):
        validate_editor_report(report, run)


def test_validate_editor_report_rejects_url_in_both_sections(run: RunRecord) -> None:
    report = EditorReport(
        top_findings=[finding("https://example.com/accepted-1")],
        watchlist=[finding("https://example.com/accepted-1")],
    )
    run.watchlist_ids.append("accepted-1")

    with pytest.raises(ReportValidationError):
        validate_editor_report(report, run)


def test_render_markdown_is_deterministic_and_complete() -> None:
    report = EditorReport(
        top_findings=[finding("https://example.com/a", 0.876)],
        watchlist=[finding("https://example.com/b", 0.724, "Watch")],
    )

    rendered = render_markdown(report, date(2026, 8, 25))

    assert "# AI Engineering Radar - 2026-08-25" in rendered
    assert "## Top Findings" in rendered
    assert "## Watchlist" in rendered
    assert "What changed" in rendered
    assert "Why it matters" in rendered
    assert "**Confidence:** 0.88" in rendered
    assert "**Skeptic's objection**" in rendered
    assert "**Source:** https://example.com/a" in rendered
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_render_markdown_empty_states() -> None:
    rendered = render_markdown(EditorReport(top_findings=[], watchlist=[]), date(2026, 8, 25))

    assert "No accepted findings in this run." in rendered
    assert "No watchlist items in this run." in rendered


def test_publish_report_creates_and_replaces_target_atomically(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports" / "nested"
    report = EditorReport(top_findings=[], watchlist=[])
    report_date = date(2026, 8, 25)

    path = publish_report(report, report_date, output_dir)

    assert path == output_dir / "2026-08-25.md"
    assert path.read_text(encoding="utf-8") == render_markdown(report, report_date)
    path.write_text("old content", encoding="utf-8")
    publish_report(report, report_date, output_dir)
    assert path.read_text(encoding="utf-8") == render_markdown(report, report_date)
    assert not list(output_dir.glob("*.tmp"))
