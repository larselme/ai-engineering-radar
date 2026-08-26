import json

from config import PROMPTS_DIR
from models.schemas import SourceItem, TriageDecision, TriageTraceStep
from agents.client import StructuredCopilotClient


def run_triage_planner(
    client: StructuredCopilotClient,
    model: str,
    item: SourceItem,
    trace: list[TriageTraceStep],
    working_notes: str,
) -> TriageDecision:
    prompt = (PROMPTS_DIR / "triage.txt").read_text(encoding="utf-8")
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    prompt += "\n\nPRIOR TRIAGE TRACE:\n" + _serialize_trace(trace)
    prompt += "\n\nWORKING NOTES:\n" + (working_notes or "(none)")
    return client.parse(model, prompt, TriageDecision)


def _serialize_trace(trace: list[TriageTraceStep]) -> str:
    if not trace:
        return "[]"
    return json.dumps([step.model_dump(mode="json") for step in trace], indent=2)
