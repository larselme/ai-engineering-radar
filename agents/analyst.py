from config import PROMPTS_DIR
from models.schemas import AnalysisResult, SourceItem, TriageTraceStep
from agents.client import StructuredCopilotClient
from agents.prompt_context import append_triage_evidence


def run_analyst(
    client: StructuredCopilotClient,
    model: str,
    item: SourceItem,
    previous_analysis: AnalysisResult | None = None,
    judge_feedback: str | None = None,
    triage_summary: str | None = None,
    triage_trace: list[TriageTraceStep] | None = None,
) -> AnalysisResult:
    prompt = (PROMPTS_DIR / "analyst.txt").read_text(encoding="utf-8")
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    prompt = append_triage_evidence(prompt, triage_summary, triage_trace)
    if previous_analysis is not None:
        prompt += "\n\nPREVIOUS ANALYSIS:\n" + previous_analysis.model_dump_json(indent=2)
    if judge_feedback is not None:
        prompt += "\n\nJUDGE FEEDBACK:\n" + judge_feedback
    return client.parse(model, prompt, AnalysisResult)
