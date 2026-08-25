import time
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


class StructuredOpenAIClient:
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key, max_retries=0)

    def parse(self, model: str, prompt: str, result_type: type[T]) -> T:
        transient_attempts = 3
        parse_attempts = 2
        last_error: Exception | None = None

        for attempt in range(transient_attempts):
            try:
                for parse_attempt in range(parse_attempts):
                    try:
                        response = self._client.responses.parse(
                            model=model,
                            input=prompt,
                            text_format=result_type,
                        )
                        parsed = getattr(response, "output_parsed", None)
                        if parsed is None:
                            raise ValueError("OpenAI response had no parsed output")
                        return result_type.model_validate(parsed)
                    except (ValidationError, ValueError) as exc:
                        last_error = exc
                        if parse_attempt == parse_attempts - 1:
                            raise
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                if attempt == transient_attempts - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Structured response parsing failed") from last_error
