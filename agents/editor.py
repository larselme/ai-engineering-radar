from config import PROMPTS_DIR
from models.schemas import CandidateRecord, EditorReport
from agents.client import StructuredOpenAIClient


def run_editor(
    client: StructuredOpenAIClient,
    model: str,
    accepted: list[CandidateRecord],
    watchlist: list[CandidateRecord],
) -> EditorReport:
    prompt = (PROMPTS_DIR / "editor.txt").read_text(encoding="utf-8")
    prompt += "\n\nACCEPTED CANDIDATES:\n" + _serialize_candidates(accepted)
    prompt += "\n\nWATCHLIST CANDIDATES:\n" + _serialize_candidates(watchlist)
    return client.parse(model, prompt, EditorReport)


def _serialize_candidates(candidates: list[CandidateRecord]) -> str:
    return "[\n" + ",\n".join(
        candidate.model_dump_json(indent=2) for candidate in candidates
    ) + "\n]"
