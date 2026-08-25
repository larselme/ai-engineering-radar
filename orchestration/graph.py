from collections.abc import Callable

from models.schemas import (
    AnalysisResult,
    CandidateRecord,
    CandidateTerminalStatus,
    CritiqueResult,
    JudgeDecision,
    JudgeStatus,
    SourceItem,
)

AnalystFn = Callable[[SourceItem, AnalysisResult | None, str | None], AnalysisResult]
SkepticFn = Callable[[SourceItem, AnalysisResult], CritiqueResult]
JudgeFn = Callable[[SourceItem, AnalysisResult, CritiqueResult, bool], JudgeDecision]


class GraphInvariantError(RuntimeError):
    """Raised when an agent attempts an impossible graph transition."""


def process_candidate(
    item: SourceItem,
    analyst: AnalystFn,
    skeptic: SkepticFn,
    judge: JudgeFn,
    max_revisions: int = 2,
) -> CandidateRecord:
    if max_revisions < 0:
        raise ValueError("max_revisions must be non-negative")

    record = CandidateRecord(source=item)

    try:
        analysis = analyst(item, None, None)
    except Exception as exc:
        record.terminal_status = CandidateTerminalStatus.ERROR
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    while True:
        record.analyses.append(analysis)

        try:
            critique = skeptic(item, analysis)
        except Exception as exc:
            record.terminal_status = CandidateTerminalStatus.ERROR
            record.error = f"{type(exc).__name__}: {exc}"
            return record
        record.critiques.append(critique)

        allow_revision = record.revision_count < max_revisions
        try:
            decision = judge(item, analysis, critique, allow_revision)
        except Exception as exc:
            record.terminal_status = CandidateTerminalStatus.ERROR
            record.error = f"{type(exc).__name__}: {exc}"
            return record
        record.decisions.append(decision)

        if decision.status is JudgeStatus.REVISE:
            if not allow_revision:
                raise GraphInvariantError(
                    "judge requested revision after the revision limit was reached"
                )
            record.revision_count += 1
            try:
                analysis = analyst(item, analysis, decision.feedback)
            except Exception as exc:
                record.terminal_status = CandidateTerminalStatus.ERROR
                record.error = f"{type(exc).__name__}: {exc}"
                return record
            continue

        record.terminal_status = CandidateTerminalStatus(decision.status.value)
        return record
