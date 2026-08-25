from config import PROMPTS_DIR
from models.schemas import AnalysisResult, SourceItem
from agents.client import StructuredOpenAIClient


def run_analyst(
    client: StructuredOpenAIClient,
    model: str,
    item: SourceItem,
    previous_analysis: AnalysisResult | None = None,
    judge_feedback: str | None = None,
) -> AnalysisResult:
    prompt = (PROMPTS_DIR / "analyst.txt").read_text(encoding="utf-8")
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    if previous_analysis is not None:
        prompt += "\n\nPREVIOUS ANALYSIS:\n" + previous_analysis.model_dump_json(indent=2)
    if judge_feedback is not None:
        prompt += "\n\nJUDGE FEEDBACK:\n" + judge_feedback
    return client.parse(model, prompt, AnalysisResult)
