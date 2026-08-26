import json
import logging
from datetime import UTC, datetime, timedelta
from functools import partial

from agents.analyst import run_analyst
from agents.client import StructuredCopilotClient
from agents.editor import run_editor
from agents.judge import run_judge
from agents.skeptic import run_skeptic
from agents.triage import run_triage_planner
from collector.collector import collect_sources
from config import LOGS_DIR, OUTPUT_DIR, STATE_DIR, RUNS_DIR, Settings, load_settings
from models.schemas import (
    CandidateRecord,
    CandidateTerminalStatus,
    RunRecord,
    RunStatus,
    SourceConfig,
    TriageTool,
)
from orchestration.graph import GraphInvariantError, process_candidate
from orchestration.triage import (
    TriageExecutionError,
    TriagePolicy,
    run_dynamic_triage,
    run_triage_tool,
)
from reporting.markdown import (
    build_editor_inputs,
    publish_report,
    validate_editor_report,
)
from storage.store import JsonStore

logger = logging.getLogger("radar")


def _configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_path = LOGS_DIR / "radar.log"
    configured_path = getattr(logger, "_radar_log_path", None)
    if configured_path != log_path:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
        logger.addHandler(logging.StreamHandler())
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        logger._radar_log_path = log_path


def _utc_now(now: datetime | None) -> tuple[datetime, bool]:
    if now is None:
        return datetime.now(UTC), False
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(UTC), True


def _load_sources() -> list[SourceConfig]:
    raw = json.loads((STATE_DIR / "sources.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("sources.json must contain a list")
    return [SourceConfig.model_validate(entry) for entry in raw]


def _default_triage_policy() -> TriagePolicy:
    return TriagePolicy(
        allowed_tools=set(TriageTool),
        max_steps=4,
        timeout_seconds=10,
        max_tokens=800,
        max_cost=0.05,
        allowed_network_scopes={"none", "official_sources"},
    )


def _bucket_candidate(
    run: RunRecord, candidate_id: str, candidate: CandidateRecord
) -> None:
    buckets = {
        CandidateTerminalStatus.ACCEPT: run.accepted_ids,
        CandidateTerminalStatus.WATCHLIST: run.watchlist_ids,
        CandidateTerminalStatus.REJECT: run.rejected_ids,
        CandidateTerminalStatus.ERROR: run.error_ids,
    }
    for bucket in buckets.values():
        while candidate_id in bucket:
            bucket.remove(candidate_id)
    if candidate.terminal_status is None:
        raise ValueError(f"candidate {candidate_id} has no terminal status")
    buckets[candidate.terminal_status].append(candidate_id)


def _markable_seen_statuses() -> set[CandidateTerminalStatus]:
    return {
        CandidateTerminalStatus.ACCEPT,
        CandidateTerminalStatus.WATCHLIST,
        CandidateTerminalStatus.REJECT,
    }


def _fail_run(
    store: JsonStore, run: RunRecord, error: Exception, end_time: datetime
) -> RunRecord:
    run.status = RunStatus.FAILED
    run.ended_at = end_time
    run.metadata["fatal_error"] = f"{type(error).__name__}: {error}"
    store.save_run(run)
    logger.error("run failure: %s", run.metadata["fatal_error"])
    return run


def run_radar(settings: Settings, now: datetime | None = None) -> RunRecord:
    run_time, deterministic_end = _utc_now(now)
    _configure_logging()
    store = JsonStore(STATE_DIR, RUNS_DIR)
    run = RunRecord(
        run_id=run_time.strftime("%Y%m%dT%H%M%SZ"),
        started_at=run_time,
    )
    store.save_run(run)
    logger.info("run start: %s", run.run_id)

    try:
        previous = store.latest_successful_run()
        since = run_time - timedelta(days=7)
        run.previous_successful_run = (
            None if previous is None else (previous.ended_at or previous.started_at)
        )
        run.metadata["collection_since"] = since.isoformat()

        logger.info("collection start")
        sources = _load_sources()
        items, source_errors = collect_sources(sources, since)
        run.source_errors = source_errors
        for source_name, error in source_errors.items():
            logger.warning("source error: %s: %s", source_name, error)
        store.save_run(run)
        logger.info("collection end: %d items", len(items))

        seen_items = store.load_seen_items()
        client = StructuredCopilotClient(
            settings.copilot_github_token,
            use_logged_in_user=settings.use_logged_in_copilot,
        )
        triage_policy = _default_triage_policy()
        triage_planner = partial(run_triage_planner, client, settings.triage_model)
        analyst = partial(run_analyst, client, settings.analyst_model)
        skeptic = partial(run_skeptic, client, settings.skeptic_model)
        judge = partial(run_judge, client, settings.judge_model)

        for item in items:
            if item.id in seen_items:
                logger.info("candidate skipped as already seen: %s", item.id)
                continue
            logger.info("candidate processing start: %s", item.id)
            candidate = store.find_reusable_terminal_candidate(item.id)
            if candidate is not None:
                logger.info("terminal candidate reused: %s", item.id)
            else:
                try:
                    triage_result = run_dynamic_triage(
                        item,
                        triage_policy,
                        triage_planner,
                        run_triage_tool,
                    )
                except TriageExecutionError as exc:
                    candidate = CandidateRecord(
                        source=item,
                        terminal_status=CandidateTerminalStatus.ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                        triage_trace=list(exc.trace),
                    )
                    logger.warning("candidate triage failure: %s (%s)", item.id, exc)
                else:
                    triage_summary = triage_result.summary
                    triage_trace = triage_result.trace

                    def triage_aware_analyst(current_item, previous_analysis, judge_feedback):
                        return analyst(
                            current_item,
                            previous_analysis,
                            judge_feedback,
                            triage_summary=triage_summary,
                            triage_trace=triage_trace,
                        )

                    def triage_aware_skeptic(current_item, current_analysis):
                        return skeptic(
                            current_item,
                            current_analysis,
                            triage_summary=triage_summary,
                            triage_trace=triage_trace,
                        )

                    def triage_aware_judge(
                        current_item,
                        current_analysis,
                        current_critique,
                        allow_revision,
                    ):
                        return judge(
                            current_item,
                            current_analysis,
                            current_critique,
                            allow_revision,
                            triage_summary=triage_summary,
                            triage_trace=triage_trace,
                        )

                    candidate = process_candidate(
                        item,
                        triage_aware_analyst,
                        triage_aware_skeptic,
                        triage_aware_judge,
                        max_revisions=settings.max_revisions,
                    )
                    candidate.triage_summary = triage_summary
                    candidate.triage_trace = triage_trace
                    logger.info(
                        "candidate triage end: %s tools=%s",
                        item.id,
                        [step.tool.value for step in triage_result.trace],
                    )

            run.candidates[item.id] = candidate
            _bucket_candidate(run, item.id, candidate)
            store.save_run(run)
            logger.info(
                "candidate processing end: %s (%s)",
                item.id,
                candidate.terminal_status,
            )

        accepted, watchlist = build_editor_inputs(run)
        logger.info("Editor start")
        report = run_editor(client, settings.editor_model, accepted, watchlist)
        logger.info("Editor end")
        validate_editor_report(report, run)
        report_path = publish_report(report, run_time.date(), OUTPUT_DIR)
        logger.info("report path: %s", report_path)

        run.status = RunStatus.SUCCESS
        run.ended_at = run_time if deterministic_end else datetime.now(UTC)
        store.save_run(run)

        seen_updates = {}
        for candidate_id, candidate in run.candidates.items():
            if candidate.terminal_status in _markable_seen_statuses():
                seen_updates[candidate_id] = {
                    "url": str(candidate.source.url),
                    "terminal_status": candidate.terminal_status.value,
                    "run_id": run.run_id,
                }
        store.mark_seen_many(seen_updates)
        logger.info("run success: %s", run.run_id)
        return run
    except GraphInvariantError as exc:
        end_time = run_time if deterministic_end else datetime.now(UTC)
        return _fail_run(store, run, exc, end_time)
    except Exception as exc:
        end_time = run_time if deterministic_end else datetime.now(UTC)
        return _fail_run(store, run, exc, end_time)


def main() -> int:
    try:
        _configure_logging()
        run = run_radar(load_settings())
    except Exception:
        logger.exception("unexpected application failure")
        return 1
    return 0 if run.status is RunStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
