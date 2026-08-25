from config import PROMPTS_DIR
from models.schemas import AnalysisResult, CritiqueResult, JudgeDecision, SourceItem
from agents.client import StructuredCopilotClient


def run_judge(
    client: StructuredCopilotClient,
    model: str,
    item: SourceItem,
    analysis: AnalysisResult,
    critique: CritiqueResult,
    allow_revision: bool,
) -> JudgeDecision:
    prompt = (PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8")
    revision_instruction = (
        "Revise is currently allowed."
        if allow_revision
        else "Revise is currently not allowed; you must not choose revise."
    )
    prompt += "\n\nREVISION POLICY:\n" + revision_instruction
    prompt += "\n\nSOURCE ITEM:\n" + item.model_dump_json(indent=2)
    prompt += "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=2)
    prompt += "\n\nSKEPTIC OUTPUT:\n" + critique.model_dump_json(indent=2)
    return client.parse(model, prompt, JudgeDecision)
