import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError, model_validator

from models.schemas import (
    SourceItem,
    TriageAction,
    TriageDecision,
    TriageOutcome,
    TriageTool,
    TriageToolResult,
    TriageTraceStep,
)

PlannerFn = Callable[[SourceItem, list[TriageTraceStep], str], TriageDecision | dict]
ToolRunnerFn = Callable[[SourceItem, TriageAction], TriageToolResult | dict]


class TriagePolicy(BaseModel):
    allowed_tools: set[TriageTool]
    max_steps: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_tokens: int = Field(ge=0)
    max_cost: float = Field(ge=0)
    allowed_network_scopes: set[str] = Field(default_factory=lambda: {"none"})

    @model_validator(mode="after")
    def validate_policy(self):
        if not self.allowed_tools:
            raise ValueError("allowed_tools must not be empty")
        if not self.allowed_network_scopes:
            raise ValueError("allowed_network_scopes must not be empty")
        return self


class TriageExecutionError(RuntimeError):
    def __init__(self, message: str, trace: list[TriageTraceStep] | None = None):
        super().__init__(message)
        self.trace = trace or []


class TriagePolicyError(TriageExecutionError):
    """Raised when the planner asks for a disallowed capability."""


class TriageTimeoutError(TriageExecutionError):
    """Raised when triage exceeds the configured timeout budget."""


class TriageBudgetError(TriageExecutionError):
    """Raised when triage exceeds token or cost budgets."""


def run_dynamic_triage(
    item: SourceItem,
    policy: TriagePolicy,
    planner: PlannerFn,
    tool_runner: ToolRunnerFn,
    *,
    now_provider: Callable[[], datetime] | None = None,
    monotonic_provider: Callable[[], float] | None = None,
) -> TriageOutcome:
    now = now_provider or (lambda: datetime.now(UTC))
    monotonic = monotonic_provider or time.monotonic

    start = monotonic()
    trace: list[TriageTraceStep] = []
    total_tokens = 0
    total_cost = 0.0
    notes = ""

    for step_number in range(1, policy.max_steps + 1):
        _guard_timeout(policy, start, monotonic, trace)

        raw_decision = planner(item, trace, notes)
        try:
            decision = TriageDecision.model_validate(raw_decision)
        except ValidationError as exc:
            raise TriageExecutionError(f"invalid triage planner output: {exc}", trace) from exc

        if decision.done:
            return TriageOutcome(
                summary=decision.final_summary,
                trace=trace,
                total_tokens=total_tokens,
                total_cost=total_cost,
            )

        action = decision.action
        if action is None:
            raise TriageExecutionError("planner omitted action while done=false", trace)

        if action.tool not in policy.allowed_tools:
            raise TriagePolicyError(
                f"tool not allowed by policy: {action.tool.value}",
                trace,
            )
        if action.network_scope not in policy.allowed_network_scopes:
            raise TriagePolicyError(
                f"network scope not allowed by policy: {action.network_scope}",
                trace,
            )

        _guard_timeout(policy, start, monotonic, trace)

        started_at = now()
        raw_result = tool_runner(item, action)
        try:
            result = TriageToolResult.model_validate(raw_result)
        except ValidationError as exc:
            raise TriageExecutionError(f"invalid triage tool result: {exc}", trace) from exc
        completed_at = now()

        step = TriageTraceStep(
            step_number=step_number,
            tool=action.tool,
            reason=action.reason,
            network_scope=action.network_scope,
            started_at=started_at,
            completed_at=completed_at,
            observation=result.observation,
            tokens_used=result.tokens_used,
            estimated_cost=result.estimated_cost,
        )
        trace.append(step)

        total_tokens += step.tokens_used
        total_cost += step.estimated_cost
        if total_tokens > policy.max_tokens:
            raise TriageBudgetError("triage token budget exceeded", trace)
        if total_cost > policy.max_cost:
            raise TriageBudgetError("triage cost budget exceeded", trace)

        notes += (
            f"\nStep {step_number} [{step.tool.value}] reason={step.reason} "
            f"observation={step.observation}"
        )

    raise TriageExecutionError("triage exhausted max_steps without completion", trace)


def run_triage_tool(item: SourceItem, action: TriageAction) -> TriageToolResult:
    content = item.content.strip()
    if action.tool is TriageTool.SUMMARIZE:
        observation = _truncate(
            f"Summary of {item.title}: {content}",
            320,
        )
    elif action.tool is TriageTool.FACT_CHECK:
        signal = "No explicit evidence found"
        lowered = content.lower()
        if "public preview" in lowered or "preview" in lowered:
            signal = "The source explicitly states this is a preview release"
        elif "ga" in lowered or "general availability" in lowered:
            signal = "The source references general availability"
        observation = signal
    elif action.tool is TriageTool.COMPARE_SOURCES:
        observation = (
            "Single-source candidate in v1. No external source expansion performed "
            "under deterministic collection boundaries."
        )
    else:
        lowered = content.lower()
        risk = "Moderate uncertainty"
        if "preview" in lowered and "no general-availability" in lowered:
            risk = "High launch-risk uncertainty due to preview without GA date"
        elif "production" in lowered:
            risk = "Lower adoption risk signaled by production language"
        observation = risk

    token_estimate = max(1, len(observation.split()))
    return TriageToolResult(
        observation=observation,
        tokens_used=token_estimate,
        estimated_cost=token_estimate * 0.00001,
    )


def _guard_timeout(
    policy: TriagePolicy,
    start: float,
    monotonic: Callable[[], float],
    trace: list[TriageTraceStep],
) -> None:
    if monotonic() - start > policy.timeout_seconds:
        raise TriageTimeoutError("triage timeout budget exceeded", trace)


def _truncate(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return text[: length - 3].rstrip() + "..."
