from datetime import UTC, datetime

import pytest

from agents.analyst import run_analyst
from agents.client import StructuredCopilotClient
from config import load_settings
from models.schemas import AnalysisClassification, AnalysisResult, SourceItem


AUTH_ERROR_HINTS = (
    "auth",
    "credential",
    "forbidden",
    "login",
    "sign in",
    "signed in",
    "token",
    "unauthorized",
)


def _is_probable_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(hint in message for hint in AUTH_ERROR_HINTS)


@pytest.mark.integration
def test_run_analyst_with_live_copilot() -> None:
    settings = load_settings()
    item = SourceItem(
        id="integration-item",
        source_name="Synthetic",
        title="Synthetic AI engineering update",
        url="https://example.com/integration-item",
        published_at=datetime(2026, 8, 25, tzinfo=UTC),
        content=(
            "Vendor X announces Tool Y in public preview. "
            "The release adds checkpoint persistence for long-running agents. "
            "No general-availability date or production reliability figures are provided."
        ),
        content_hash="integration-hash",
    )
    client = StructuredCopilotClient(
        settings.copilot_github_token,
        use_logged_in_user=settings.use_logged_in_copilot,
    )

    try:
        result = run_analyst(client, settings.analyst_model, item)
    except Exception as exc:
        if _is_probable_auth_error(exc):
            pytest.skip(f"GitHub Copilot authentication unavailable: {exc}")
        raise

    assert isinstance(result, AnalysisResult)
    assert 0 <= result.confidence <= 1
    assert result.classification in set(AnalysisClassification)