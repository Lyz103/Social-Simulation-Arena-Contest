"""Task routing, calibrated uncertainty, and immutable run identity contracts."""

from dataclasses import replace

import pytest

from ssa_harness.archive import RunArchive
from ssa_harness.contracts import ForecastRequest
from ssa_harness.ensemble import BaselineFactory, EnsembleEngine, ForecastMember
from ssa_harness.llm import ReplayClient
from ssa_harness.pipeline import ForecastPipeline, PipelineConfig
from ssa_harness.retrieval.errors import ArchiveConflict
from ssa_harness.retrieval.utils import digest


def request(**changes):
    values = dict(
        round_id="policy-round",
        question="Forecast next approval",
        target_type="continuous_normal",
        unit="%",
        source_name="Source A",
        population="adults",
        metric="approval",
        asof="2026-09-20T00:00:00Z",
        deadline="2026-09-21T00:00:00Z",
        history=(
            {"date": "2026-09-17", "available_at": "2026-09-18T00:00:00Z", "value": 40},
            {"date": "2026-09-18", "available_at": "2026-09-19T00:00:00Z", "value": 41},
            {"date": "2026-09-19", "available_at": "2026-09-19T12:00:00Z", "value": 43},
        ),
    )
    values.update(changes)
    return ForecastRequest(**values)


def pipeline(tmp_path, *, policies=(), baseline=None, scale=1.0):
    return ForecastPipeline(
        archive=RunArchive(tmp_path / "runs.sqlite3"),
        clients={"fixture": ReplayClient({"policy-round": {"mean": 44, "sd": 2}})},
        config=PipelineConfig(entrant="policy-test", research_policy="disabled", sd_scale=scale),
        ensemble=EnsembleEngine(task_policies=policies, baseline_factory=baseline),
    )


def test_defaults_preserve_existing_prediction():
    req = request()
    old = EnsembleEngine().combine(req, [{"mean": 44, "sd": 2}])
    unmatched = EnsembleEngine(task_policies=[
        {"match": {"source_name": "Source B"}, "weights": {"persistence": 1}, "sd_scale": 5}
    ]).combine(req, [{"mean": 44, "sd": 2}])
    assert old.prediction == unmatched.prediction
    assert old.weights == unmatched.weights
    assert unmatched.policy["index"] is None


def test_most_specific_exact_match_overrides_weights_completely():
    engine = EnsembleEngine(task_policies=[
        {"match": {"target_type": "continuous_normal"}, "weights": {"llm": 1}},
        {"match": {"source_name": "Source A", "metric": "approval"}, "weights": {"persistence": 1}},
    ])
    final = engine.combine(request(), [{"mean": 1e308, "sd": 2}])
    assert final.prediction == {"mean": 43.0, "sd": 0.5}
    assert final.weights == {"llm-1": 0.0, "persistence": 1.0, "recent_mean": 0.0}
    assert not final.degraded
    assert final.policy["index"] == 1
    assert engine.resolve_policy(request(source_name="source a"))["index"] == 0


@pytest.mark.parametrize("field,value", [
    ("target_type", "continuous_normal"), ("source_name", "Source A"),
    ("metric", "approval"), ("population", "adults"), ("unit", "%"),
])
def test_allowed_fields_match_exactly(field, value):
    engine = EnsembleEngine(task_policies=[{"match": {field: value}, "sd_scale": 2}])
    assert engine.resolve_policy(request())["policy_sd_scale"] == 2


def test_equal_specificity_matching_policies_are_ambiguous():
    engine = EnsembleEngine(task_policies=[
        {"match": {"target_type": "continuous_normal"}, "sd_scale": 2},
        {"match": {"source_name": "Source A"}, "sd_scale": 3},
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        engine.combine(request(), [{"mean": 44, "sd": 2}])


@pytest.mark.parametrize("match", [{"round_id": "policy-round"}, {"metadata": "x"}, {"question": "x"}])
def test_outcome_or_question_specific_routing_is_rejected(match):
    with pytest.raises(ValueError, match="allowed exact string fields"):
        EnsembleEngine(task_policies=[{"match": match, "weights": {"persistence": 1}}])


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_invalid_policy_scales_are_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        EnsembleEngine(task_policies=[{"match": {}, "sd_scale": scale}])


def test_explicit_zero_llm_weight_with_no_llm_is_not_a_failure():
    final = EnsembleEngine(task_policies=[
        {"match": {"unit": "%"}, "weights": {"llm": 0, "persistence": 1}}
    ]).combine(request(), [])
    assert final.prediction["mean"] == 43
    assert not final.degraded


def test_engine_scales_each_profile_cell_once_without_changing_members():
    req = request(target_type="profile_energy", cells=("a", "b"), history=())
    member = ForecastMember("llm", "llm", {"cells": {
        "a": {"mean": 10, "sd": 2}, "b": {"mean": 20, "sd": 3}
    }})
    engine = EnsembleEngine(task_policies=[
        {"match": {"target_type": "profile_energy"}, "weights": {"llm": 1}, "sd_scale": 2}
    ])
    final = engine.combine(req, [member], baselines=[], sd_scale=3)
    assert final.prediction["cells"]["a"]["sd"] == 12
    assert final.prediction["cells"]["b"]["sd"] == 18
    assert member.prediction["cells"]["a"]["sd"] == 2
    assert final.policy["effective_sd_scale"] == 6


def test_pipeline_scales_once_and_archives_effective_policy(tmp_path):
    policies = [{"match": {"source_name": "Source A"}, "weights": {"llm": 1}, "sd_scale": 2}]
    runner = pipeline(tmp_path, policies=policies, scale=3)
    result = runner.run(request())
    assert result["response"]["forecast"]["sd"] == 12
    assert result["final"]["policy"]["effective_sd_scale"] == 6
    assert result["final"]["policy"]["global_sd_scale"] == 3
    assert result["final"]["policy"]["policy_sd_scale"] == 2
    assert result["pipeline"]["ensemble"]["task_policies"] == runner.ensemble.task_policies
    assert result["pipeline_config_hash"] == digest(runner.identity())
    assert runner.run(request()) == result
    assert runner.clients["fixture"].calls == 1
    changed = pipeline(tmp_path, policies=[{**policies[0], "sd_scale": 4}], scale=3)
    with pytest.raises(ArchiveConflict):
        changed.run(request())
    assert changed.clients["fixture"].calls == 0


def test_all_baseline_configuration_and_global_scale_change_identity(tmp_path):
    original = pipeline(tmp_path)
    base_hash = digest(original.identity())
    baseline = BaselineFactory(ewma_alpha=0.5)
    changed = pipeline(tmp_path, baseline=baseline)
    assert digest(changed.identity()) != base_hash
    assert changed.identity()["ensemble"]["baseline"] == baseline.identity()
    original.config = replace(original.config, sd_scale=2)
    assert digest(original.identity()) != base_hash


def test_cli_calibration_objective_is_forwarded(tmp_path, monkeypatch):
    import json
    from ssa_harness import cli

    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    called = {}

    def calibrate(cases, **kwargs):
        called.update(kwargs)
        return {"objective": kwargs["objective"]}

    monkeypatch.setattr("ssa_harness.evaluation.calibrate", calibrate)
    output = tmp_path / "result.json"
    cli.main(["calibrate", "--cases", str(cases), "--validation-start", "2026-08-01",
              "--test-start", "2026-09-01", "--objective", "mean_skill", "--output", str(output)])
    assert called["objective"] == "mean_skill"
    assert json.loads(output.read_text()) == {"objective": "mean_skill"}
