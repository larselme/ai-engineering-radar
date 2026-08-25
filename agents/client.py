import asyncio
import json
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from copilot import CopilotClient
from copilot.session import PermissionHandler

T = TypeVar("T", bound=BaseModel)


class CopilotStructuredOutputError(RuntimeError):
    pass


class StructuredCopilotClient:
    def __init__(
        self,
        github_token: str | None = None,
        *,
        use_logged_in_user: bool = True,
    ):
        self._github_token = github_token
        self._use_logged_in_user = use_logged_in_user

    def parse(self, model: str, prompt: str, result_type: type[T]) -> T:
        transient_attempts = 3
        parse_attempts = 2
        last_error: Exception | None = None
        non_transient_errors = (
            ValidationError,
            ValueError,
            CopilotStructuredOutputError,
        )

        for attempt in range(transient_attempts):
            try:
                for parse_attempt in range(parse_attempts):
                    try:
                        response_text = asyncio.run(
                            self._send_prompt(model, prompt, result_type)
                        )
                        parsed = self._extract_json_payload(response_text)
                        return result_type.model_validate(parsed)
                    except (ValidationError, ValueError) as exc:
                        last_error = exc
                        if parse_attempt == parse_attempts - 1:
                            raise
            except non_transient_errors:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == transient_attempts - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Structured response parsing failed") from last_error

    async def _send_prompt(self, model: str, prompt: str, result_type: type[T]) -> str:
        async with CopilotClient(
            github_token=self._github_token,
            use_logged_in_user=self._use_logged_in_user,
        ) as client:
            async with await client.create_session(
                on_permission_request=PermissionHandler.approve_all,
                model=model,
                infinite_sessions={"enabled": False},
                system_message={
                    "mode": "append",
                    "content": self._structured_output_instruction(result_type),
                },
            ) as session:
                response = await session.send_and_wait(prompt)
                if response is None or getattr(response, "data", None) is None:
                    raise CopilotStructuredOutputError(
                        "Copilot response had no message data"
                    )
                content = getattr(response.data, "content", None)
                if not content:
                    raise CopilotStructuredOutputError(
                        "Copilot response had no content"
                    )
                return content

    @staticmethod
    def _structured_output_instruction(result_type: type[T]) -> str:
        schema = json.dumps(result_type.model_json_schema(), indent=2)
        return (
            "Return only valid JSON matching this schema. Do not include markdown, "
            "comments, or explanatory text.\n\nJSON SCHEMA:\n"
            f"{schema}"
        )

    @staticmethod
    def _extract_json_payload(content: str) -> dict:
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError("Copilot response did not contain valid JSON")
            return json.loads(text[start : end + 1])
