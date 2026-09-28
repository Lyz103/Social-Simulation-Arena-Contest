import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/validate_arena.py"
    spec = importlib.util.spec_from_file_location("arena_validation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA = tmp_path
    raw = {
        "round_id": "export-test", "question": "Next value", "target_type": "continuous_normal",
        "unit": "points", "asof": "2026-08-01T00:00:00Z", "deadline": "2026-08-02T00:00:00Z",
        "history": [{"date": "2026-07-01", "available_at": "2026-07-02T00:00:00Z", "value": 10}],
    }
    module.write(tmp_path / "inputs.json", [raw])
    module.write(tmp_path / "labels.json", {"export-test": {"truth": 11, "persistence": {"mean": 10, "sd": 1.5}}})
    return module


def test_prediction_stage_never_reads_labels_and_cannot_replace_forecast(runner, monkeypatch):
    original = runner.read
    def checked(path):
        assert Path(path).name != "labels.json"
        return original(path)
    monkeypatch.setattr(runner, "read", checked)
    runner.predict()
    f = runner.DATA / "predictions/baseline/export-test.json"
    saved = original(f)
    assert saved["input_hash"] and saved["config_hash"]
    with pytest.raises(ValueError, match="prediction differs"):
        runner.write_prediction(f, {**saved, "input_hash": "wrong"})


def test_score_rejects_changed_inputs(runner):
    runner.predict()
    f = runner.DATA / "inputs.json"
    inputs = runner.read(f)
    inputs[0]["question"] = "Different target"
    runner.write(f, inputs)
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.score()


def test_score_rejects_duplicate_strategy_files(runner):
    runner.predict()
    f = runner.DATA / "predictions/baseline/export-test.json"
    runner.write(runner.DATA / "predictions/duplicate/export-test.json", runner.read(f))
    with pytest.raises(ValueError, match="duplicate prediction strategies"):
        runner.score()


def test_legacy_upgrade_recomputes_baselines_before_binding_identity(runner):
    runner.predict()
    f = runner.DATA / "predictions/baseline/export-test.json"
    expected = runner.read(f)
    legacy = {"round_id": expected["round_id"], "predictions": expected["predictions"]}
    runner.write(f, legacy)
    runner.verify_legacy_exports()
    assert runner.read(f) == expected
    legacy["predictions"]["persistence"]["mean"] = 999
    runner.write(f, legacy)
    with pytest.raises(ValueError, match="failed verification"):
        runner.verify_legacy_exports()
