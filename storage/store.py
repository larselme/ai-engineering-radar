import json
from collections.abc import Iterator
from pathlib import Path

from models.schemas import CandidateRecord, RunRecord, RunStatus


class JsonStore:
    def __init__(self, state_dir: Path, runs_dir: Path):
        self.state_dir = state_dir
        self.runs_dir = runs_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.seen_items_path = self.state_dir / "seen_items.json"

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    def load_seen_items(self) -> dict[str, dict]:
        if not self.seen_items_path.exists():
            return {}
        with self.seen_items_path.open(encoding="utf-8") as file:
            return json.load(file)

    def mark_seen_many(self, updates: dict[str, dict]) -> None:
        if not updates:
            return
        seen_items = self.load_seen_items()
        seen_items.update(updates)
        self._write_atomic(
            self.seen_items_path,
            json.dumps(seen_items, indent=2),
        )

    def mark_seen(self, candidate_id: str, payload: dict) -> None:
        self.mark_seen_many({candidate_id: payload})

    def save_run(self, run: RunRecord) -> Path:
        path = self.runs_dir / f"{run.run_id}.json"
        self._write_atomic(path, run.model_dump_json(indent=2))
        return path

    def load_run(self, path: Path) -> RunRecord:
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_runs(self) -> Iterator[RunRecord]:
        return (self.load_run(path) for path in self.runs_dir.glob("*.json"))

    def latest_successful_run(self) -> RunRecord | None:
        successful_runs = (
            run for run in self.iter_runs() if run.status is RunStatus.SUCCESS
        )
        return max(successful_runs, key=lambda run: run.started_at, default=None)

    def find_reusable_terminal_candidate(
        self,
        candidate_id: str,
    ) -> CandidateRecord | None:
        runs = sorted(self.iter_runs(), key=lambda run: run.started_at, reverse=True)
        for run in runs:
            candidate = run.candidates.get(candidate_id)
            if candidate is not None and candidate.terminal_status is not None:
                return candidate
        return None
