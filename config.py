import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
RUNS_DIR = ROOT_DIR / "runs"
OUTPUT_DIR = ROOT_DIR / "output"
LOGS_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "prompts"

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    openai_api_key: str
    analyst_model: str = "gpt-5.6-luna"
    skeptic_model: str = "gpt-5.6-luna"
    judge_model: str = "gpt-5.6-luna"
    editor_model: str = "gpt-5.6-luna"
    max_revisions: int = 2


def load_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    max_revisions_raw = os.getenv("MAX_REVISIONS", "2")
    try:
        max_revisions = int(max_revisions_raw)
    except ValueError as exc:
        raise ValueError("MAX_REVISIONS must be 2 for v1") from exc

    if max_revisions != 2:
        raise ValueError("MAX_REVISIONS must be 2 for v1")

    return Settings(
        openai_api_key=openai_api_key,
        analyst_model=os.getenv("ANALYST_MODEL", "gpt-5.6-luna"),
        skeptic_model=os.getenv("SKEPTIC_MODEL", "gpt-5.6-luna"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-5.6-luna"),
        editor_model=os.getenv("EDITOR_MODEL", "gpt-5.6-luna"),
        max_revisions=max_revisions,
    )
