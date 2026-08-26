from models.schemas import TriageOutcome, TriageTraceStep


def append_triage_evidence(
    prompt: str,
    triage_summary: str | None,
    triage_trace: list[TriageTraceStep] | None,
) -> str:
    summary = (triage_summary or "").strip()
    trace = triage_trace or []
    if not summary and not trace:
        return prompt
    evidence = TriageOutcome(summary=summary, trace=trace)
    return prompt + "\n\nTRIAGE EVIDENCE:\n" + evidence.model_dump_json(indent=2)
