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

    copilot_github_token: str | None = None
    use_logged_in_copilot: bool = True
    analyst_model: str = "gpt-5.4"
    skeptic_model: str = "gpt-5.4"
    judge_model: str = "gpt-5.4"
    editor_model: str = "gpt-5.4"
    max_revisions: int = 2


def load_settings() -> Settings:
    copilot_github_token = os.getenv("COPILOT_GITHUB_TOKEN")
    use_logged_in_copilot = os.getenv("USE_LOGGED_IN_COPILOT", "true").lower() not in {
        "0",
        "false",
        "no",
    }

    if not use_logged_in_copilot and not copilot_github_token:
        raise ValueError(
            "COPILOT_GITHUB_TOKEN is required when USE_LOGGED_IN_COPILOT is false"
        )

    max_revisions_raw = os.getenv("MAX_REVISIONS", "2")
    try:
        max_revisions = int(max_revisions_raw)
    except ValueError as exc:
        raise ValueError("MAX_REVISIONS must be 2 for v1") from exc

    if max_revisions != 2:
        raise ValueError("MAX_REVISIONS must be 2 for v1")

    return Settings(
        copilot_github_token=copilot_github_token,
        use_logged_in_copilot=use_logged_in_copilot,
        analyst_model=os.getenv("ANALYST_MODEL", "gpt-5.4"),
        skeptic_model=os.getenv("SKEPTIC_MODEL", "gpt-5.4"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-5.4"),
        editor_model=os.getenv("EDITOR_MODEL", "gpt-5.4"),
        max_revisions=max_revisions,
    )
