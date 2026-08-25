# Copilot instructions

## Project overview

This repository is the foundation for an AI engineering radar. It currently
defines the domain contracts and durable run state; there is not yet a CLI or
orchestration pipeline.

- `models/schemas.py` contains the Pydantic v2 domain model layer. It defines
  source configuration/items, analyst/critic/judge outputs, candidate state,
  editor output, and run status.
- `config.py` is the single configuration entry point. It loads `.env` from
  the repository root and exposes frozen `Settings`, including the OpenAI key,
  per-stage model names, and the currently fixed `max_revisions` value.
- `storage/store.py` provides `JsonStore`, the persistence boundary for
  `state/seen_items.json` and per-run JSON files under `runs/`. Writes are
  atomic, and loaded run files are validated back into `RunRecord` objects.
- `tests/` contains pytest coverage for schema validation and store behavior.
  Use `tmp_path` for store tests so tests never modify repository state.
- `logs/`, `runs/`, `output/`, and the seen-items state file are runtime
  locations. Generated files in these locations are ignored by Git (apart
  from the directory placeholders and the tracked initial state file).

## Setup and commands

The project uses Python dependencies from `requirements.txt` and has no
separate build or lint configuration at present.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Run the complete test suite:

```powershell
py -m pytest
```

Run one test module, one test function, or one parametrized case:

```powershell
py -m pytest tests/test_state.py
py -m pytest tests/test_state.py::test_save_run_and_load_run_round_trip
py -m pytest tests/test_schemas.py -k confidence
```

## Repository conventions

- Use Pydantic v2 APIs (`model_dump_json`, `model_validate_json`,
  `model_validator`) and preserve typed models as the boundary between
  pipeline stages.
- Reuse the existing `StrEnum` status/classification enums rather than
  introducing free-form status strings.
- Put validation constraints on the model fields. Confidence values are
  unit-interval values; revision counts are non-negative; judge revisions
  require non-empty feedback; report lists are capped at five items.
- Use `Field(default_factory=...)` for mutable list/dict fields.
- Treat `RunRecord` status and candidate terminal status as distinct:
  a run can fail while containing reusable terminal candidates.
- Persist through `JsonStore` rather than writing state/run JSON directly.
  Preserve atomic replacement semantics when changing persistence code.
- Resolve repository data paths through the constants in `config.py`; do not
  assume the process current directory is the repository root.
- Keep API credentials in `.env` (based on `.env.example`); never commit
  `.env` or secrets. `OPENAI_API_KEY` is required, and `MAX_REVISIONS` must
  remain `2` until the v1 configuration contract changes.
- Tests use timezone-aware UTC datetimes and compare validated model instances,
  so new date/time behavior should preserve timezone awareness and round-trip
  validation.
