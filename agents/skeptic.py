from config import PROMPTS_DIR
from models.schemas import AnalysisResult, CritiqueResult, SourceItem, TriageTraceStep
from agents.client import StructuredCopilotClient
from agents.prompt_context import append_triage_evidence


def run_skeptic(
    client: StructuredCopilotClient,
    model: str,
    item: SourceItem,
    analysis: AnalysisResult,
    triage_summary: str | None = None,
    triage_trace: list[TriageTraceStep] | None = None,
) -> CritiqueResult:
    prompt = (PROMPTS_DIR / "skeptic.txt").read_text(encoding="utf-8")
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    prompt = append_triage_evidence(prompt, triage_summary, triage_trace)
    prompt += "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=2)
    return client.parse(model, prompt, CritiqueResult)
