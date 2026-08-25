import pytest


@pytest.fixture(autouse=True)
def isolate_radar_runtime(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path / "logs")
