from config import PROMPTS_DIR
from models.schemas import AnalysisResult, CritiqueResult, SourceItem
from agents.client import StructuredOpenAIClient


def run_skeptic(
    client: StructuredOpenAIClient,
    model: str,
    item: SourceItem,
    analysis: AnalysisResult,
) -> CritiqueResult:
    prompt = (PROMPTS_DIR / "skeptic.txt").read_text(encoding="utf-8")
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    prompt += "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=2)
    return client.parse(model, prompt, CritiqueResult)
