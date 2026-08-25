from datetime import date
from pathlib import Path

from models.schemas import (
    CandidateRecord,
    CandidateTerminalStatus,
    EditorFinding,
    EditorReport,
    RunRecord,
)


class ReportValidationError(ValueError):
    """Raised when an editor report violates its provenance boundary."""


def build_editor_inputs(
    run: RunRecord,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    accepted = _candidates_for_ids(
        run, run.accepted_ids, CandidateTerminalStatus.ACCEPT
    )
    watchlist = _candidates_for_ids(
        run, run.watchlist_ids, CandidateTerminalStatus.WATCHLIST
    )
    return accepted, watchlist


def _candidates_for_ids(
    run: RunRecord,
    candidate_ids: list[str],
    expected_status: CandidateTerminalStatus,
) -> list[CandidateRecord]:
    return [
        candidate
        for candidate_id in candidate_ids
        if (candidate := run.candidates.get(candidate_id)) is not None
        and candidate.terminal_status is expected_status
    ]


def validate_editor_report(report: EditorReport, run: RunRecord) -> None:
    accepted_urls = _urls_for_ids(run, run.accepted_ids, CandidateTerminalStatus.ACCEPT)
    watchlist_urls = _urls_for_ids(
        run, run.watchlist_ids, CandidateTerminalStatus.WATCHLIST
    )
    rejected_or_error_urls = _urls_for_ids(
        run,
        [*run.rejected_ids, *run.error_ids],
        None,
    )
    known_urls = {
        str(candidate.source.url)
        for candidate in run.candidates.values()
    }

    top_urls = _validate_section_urls(
        report.top_findings,
        accepted_urls,
        "top_findings",
        known_urls,
        rejected_or_error_urls,
    )
    watchlist_urls_in_report = _validate_section_urls(
        report.watchlist,
        watchlist_urls,
        "watchlist",
        known_urls,
        rejected_or_error_urls,
    )

    if top_urls & watchlist_urls_in_report:
        raise ReportValidationError(
            "a source URL must not appear in both top_findings and watchlist"
        )


def _urls_for_ids(
    run: RunRecord,
    candidate_ids: list[str],
    expected_status: CandidateTerminalStatus | None,
) -> set[str]:
    urls: set[str] = set()
    for candidate_id in candidate_ids:
        candidate = run.candidates.get(candidate_id)
        if candidate is not None and (
            expected_status is None or candidate.terminal_status is expected_status
        ):
            urls.add(str(candidate.source.url))
    return urls


def _validate_section_urls(
    findings: list[EditorFinding],
    allowed_urls: set[str],
    section_name: str,
    known_urls: set[str],
    rejected_or_error_urls: set[str],
) -> set[str]:
    seen: set[str] = set()
    for finding in findings:
        url = str(finding.source_url)
        if url not in known_urls:
            raise ReportValidationError(f"unknown source URL in {section_name}: {url}")
        if url in rejected_or_error_urls:
            raise ReportValidationError(
                f"rejected or error source URL in {section_name}: {url}"
            )
        if url not in allowed_urls:
            raise ReportValidationError(
                f"source URL is not allowed in {section_name}: {url}"
            )
        if url in seen:
            raise ReportValidationError(
                f"duplicate source URL in {section_name}: {url}"
            )
        seen.add(url)
    return seen


def render_markdown(report: EditorReport, report_date: date) -> str:
    lines = [
        f"# AI Engineering Radar - {report_date:%Y-%m-%d}",
        "",
        "## Top Findings",
        "",
    ]
    if report.top_findings:
        for index, finding in enumerate(report.top_findings, start=1):
            lines.extend(_render_finding(finding, f"## {index}."))
    else:
        lines.append("No accepted findings in this run.")

    lines.extend(["", "## Watchlist", ""])
    if report.watchlist:
        for finding in report.watchlist:
            lines.extend(_render_finding(finding, "###"))
    else:
        lines.append("No watchlist items in this run.")
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_finding(finding: EditorFinding, heading_prefix: str) -> list[str]:
    return [
        f"{heading_prefix} {finding.title}",
        "",
        "**What changed**",
        "",
        finding.what_changed,
        "",
        "**Why it matters**",
        "",
        finding.why_it_matters,
        "",
        f"**Confidence:** {finding.confidence:.2f}",
        "",
        "**Skeptic's objection**",
        "",
        finding.skeptic_objection,
        "",
        f"**Source:** {finding.source_url}",
        "",
    ]


def publish_report(
    report: EditorReport,
    report_date: date,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{report_date:%Y-%m-%d}.md"
    temporary = target.with_name(f".{target.name}.tmp")
    content = render_markdown(report, report_date)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
