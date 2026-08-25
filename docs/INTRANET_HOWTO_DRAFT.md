# AI Engineering Radar: How We Built a Small Autonomous Agent System

## Why We Built It

We wanted a deliberately small system to learn practical agent engineering end to
end, not just prompt engineering in isolation. The project goals were:

- Build a bounded autonomous flow with explicit control and stopping rules
- Generate a useful, auditable output for internal AI-engineering awareness
- Keep orchestration understandable by reading code in one sitting

## What the System Does

The radar scans official AI-engineering sources, processes each candidate through
four roles, and publishes a deterministic Markdown report.

Role responsibilities:

- Analyst: summarize and classify relevance
- Skeptic: challenge claims and identify weak evidence
- Judge: decide `accept`, `watchlist`, `reject`, or `revise`
- Editor: rank accepted/watchlist items for publication

Python owns topology, retries, persistence, and invariants. Agents are bounded to
judgment inside that control envelope.

## Core Design Decisions

1. Explicit graph, no hidden orchestration runtime.
2. Bounded revision loop (`max_revisions = 2`) for predictable execution.
3. Atomic JSON persistence for run records and seen-state.
4. Strict provenance validation before report publication.
5. Deterministic output rendering in Python, not model-formatted text.

## Reliability Changes That Mattered

Two lifecycle corrections were decisive:

1. Rolling collection window

- Earlier behavior used previous successful run as collection cutoff.
- That could miss content when a source was unavailable during an otherwise
  successful run.
- Fixed behavior: every run scans exactly the previous 7 days.

2. Batch seen-state mutation

- Earlier behavior marked seen items one by one.
- A late write failure could leave partial state.
- Fixed behavior: all seen updates are merged and atomically written once.

## Provider Migration: OpenAI -> GitHub Copilot

We migrated provider usage to GitHub Copilot while preserving existing agent
contracts (`parse(model, prompt, result_type)`).

What changed:

- Runtime client moved to Copilot SDK session calls
- Structured output is enforced by schema instruction + JSON validation
- Existing orchestration and domain schemas remained unchanged

Why this worked:

- The provider boundary was narrow (`agents/client.py`)
- Domain contracts were already strongly typed with Pydantic

## TLS/Network Lesson

In a corporate network, source collection initially failed due to certificate trust.
Using the system trust store in the collector HTTP client resolved this without
disabling TLS verification.

Principle:

- Keep TLS verification enabled
- Integrate with enterprise trust roots correctly
- Never use `verify=False` in production code

## Testing Approach

Testing strategy that proved effective:

- Fast unit tests for schemas, graph routing, persistence, reporting, collector parsing
- Lifecycle tests for end-to-end run behavior and failure semantics
- One live provider integration test (marker-gated) for real API path validation

This gave confidence without making normal CI dependent on external services.

## What We Learned

1. A green test suite can still miss intent if tests are not truly exercising the
   path they claim.
2. Component correctness is not enough; lifecycle composition creates new failure
   modes.
3. Explicit invariants and atomic writes remove whole classes of production bugs.
4. Small, bounded agent systems are easier to trust, debug, and explain.

## Current Operational Flow

- Schedule runner starts `main.py`
- Sources are collected deterministically
- Candidates run through Analyst -> Skeptic -> Judge with bounded revisions
- Editor composes publishable findings
- Python validates provenance and publishes deterministic Markdown
- Seen-state is atomically updated once on success

## Suggested Rollout Guidance

For teams adopting this pattern:

1. Start with strict boundaries and minimal moving parts.
2. Treat provenance validation as mandatory, not optional.
3. Add one real integration test per external dependency.
4. Prioritize recovery semantics (resume, dedupe, idempotence) early.

## References in Repo

- `main.py`
- `orchestration/graph.py`
- `storage/store.py`
- `reporting/markdown.py`
- `collector/http.py`
- `tests/test_run_lifecycle.py`