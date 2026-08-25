# AI Engineering Radar

A small autonomous multi-agent radar that scans official AI-engineering sources,
runs a bounded Analyst -> Skeptic -> Judge loop per candidate, and publishes a
deterministic Markdown report.

## What It Produces

Each successful run produces:

- Top findings (up to 5)
- Watchlist (up to 5)
- Source URL, what changed, why it matters, confidence, and surviving skeptic objection
- Durable run history and seen-state for safe reruns

## Architecture

```text
Collector
   |
   v
Analyst -> Skeptic -> Judge
  ^                  |
  |------ REVISE ----|
                     |
          ACCEPT / WATCHLIST / REJECT
                     |
                   Editor
                     |
             deterministic Markdown
```

## Repository Structure

- `main.py`: run lifecycle and orchestration wiring
- `collector/`: deterministic RSS/webpage collection
- `agents/`: bounded LLM role contracts and provider client
- `orchestration/`: explicit revision-bounded routing graph
- `reporting/`: report validation and deterministic Markdown rendering
- `storage/`: atomic JSON persistence and recovery
- `models/`: domain schemas and validation
- `state/`: source config and seen-item state
- `runs/`: run records
- `output/`: published daily reports

## Prerequisites

- Windows + PowerShell
- Python 3.13
- GitHub Copilot CLI installed and authenticated, or `COPILOT_GITHUB_TOKEN`

## Setup (Windows)

```powershell
git clone https://github.com/larselme/ai-engineering-radar.git
cd ai-engineering-radar

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

## Configuration

Edit `.env`:

- `USE_LOGGED_IN_COPILOT=true` (default) uses local Copilot login
- or set `COPILOT_GITHUB_TOKEN` and `USE_LOGGED_IN_COPILOT=false`
- choose models with `ANALYST_MODEL`, `SKEPTIC_MODEL`, `JUDGE_MODEL`, `EDITOR_MODEL`

`MAX_REVISIONS` is fixed to `2` for v1.

## Run

```powershell
py -3.13 main.py
```

or:

```powershell
scripts\run-radar.cmd
```

Outputs:

- Report: `output\YYYY-MM-DD.md`
- Run record: `runs\<RUN-ID>.json`
- Log: `logs\radar.log`

## Tests

Run all non-integration tests:

```powershell
py -3.13 -m pytest -v -m "not integration"
```

Run live Copilot integration test:

```powershell
py -3.13 -m pytest tests/test_integration_copilot.py -v -m integration
```

## Notes

- Source collection uses the system trust store (via `truststore`) when available.
- Collector source failures are non-fatal; the run can still succeed.
- Every run scans the previous 7 days; seen-state dedupe prevents repeat processing.