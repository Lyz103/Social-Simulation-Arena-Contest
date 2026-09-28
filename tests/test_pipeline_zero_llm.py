"""An intentional statistical policy must not depend on an unused model API."""

import pytest

from ssa_harness.archive import RunArchive
from ssa_harness.contracts import ForecastRequest
from ssa_harness.ensemble import EnsembleEngine
from ssa_harness.llm import ModelError
from ssa_harness.pipeline import ForecastPipeline, PipelineConfig


class FailedClient:
    def __init__(self):
        self.calls = 0

    def identity(self):
        return {"model": "local-failure-fixture"}

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise ModelError("simulated unavailable model")


def request():
    return ForecastRequest(
        round_id="zero-llm-round",
        question="Next approval rate",
        target_type="continuous_normal",
        unit="points",
        source_name="Source A",
        metric="approval",
        asof="2026-01-02T12:00:00Z",
        deadline="2026-01-03T12:00:00Z",
        history=(
            {"date": "2026-01-01", "available_at": "2026-01-02T00:00:00Z", "value": 10},
        ),
    )


def pipeline(tmp_path, *, weights=None, policies=(), allow_fallback=False):
    client = FailedClient()
    runner = ForecastPipeline(
        archive=RunArchive(tmp_path / "runs.sqlite3"),
        clients={"unused-model": client},
        config=PipelineConfig(
            entrant="zero-llm-test",
            research_policy="disabled",
            allow_baseline_fallback=allow_fallback,
        ),
        ensemble=EnsembleEngine(weights=weights, task_policies=policies),
    )
    return runner, client


@pytest.mark.parametrize("weights,policies,policy_index", [
    ({"llm": 0, "persistence": 1}, [], None),
    ({"persistence": 1}, [], None),
    (None, [{"match": {"source_name": "Source A"},
             "weights": {"llm": 0, "persistence": 1}}], 0),
    (None, [{"match": {"source_name": "Source A"},
             "weights": {"persistence": 1}}], 0),
])
def test_zero_selected_llm_weight_skips_api_and_is_not_degraded(
    tmp_path, weights, policies, policy_index
):
    runner, client = pipeline(tmp_path, weights=weights, policies=policies)
    result = runner.run(request())
    assert client.calls == 0
    assert result["model_runs"] == []
    assert result["member_failures"] == []
    assert result["final"]["prediction"] == {"mean": 10, "sd": 0.25}
    assert not result["final"]["degraded"]
    skips = [
        event["data"]
        for event in runner.archive.events("zero-llm-test", "zero-llm-round")
        if event["stage"] == "model_skip"
    ]
    assert len(skips) == 1
    assert "zero LLM weight" in skips[0]["reason"]
    assert skips[0]["policy_index"] == policy_index
    assert skips[0]["member_ids"] == ["unused-model"]
    assert runner.run(request()) == result
    assert client.calls == 0


@pytest.mark.parametrize("weights,policies", [
    (None, []),
    ({"llm": 0, "persistence": 1}, [
        {"match": {"source_name": "Source A"}, "weights": {"llm": 1}}
    ]),
    (None, [{"match": {"source_name": "Source B"}, "weights": {"persistence": 1}}]),
])
def test_nonzero_selected_llm_weight_retains_failure_protection(tmp_path, weights, policies):
    runner, client = pipeline(tmp_path, weights=weights, policies=policies)
    with pytest.raises(ModelError, match="baseline fallback is disabled"):
        runner.run(request())
    assert client.calls == 1
    assert not any(
        event["stage"] == "model_skip"
        for event in runner.archive.events("zero-llm-test", "zero-llm-round")
    )


def test_nonzero_llm_failure_with_explicit_fallback_remains_degraded(tmp_path):
    runner, client = pipeline(tmp_path, allow_fallback=True)
    result = runner.run(request())
    assert client.calls == 1
    assert result["final"]["degraded"]
    assert result["member_failures"][0]["member_id"] == "unused-model"


def test_failed_transport_attempts_are_archived(tmp_path):
    runner, client = pipeline(tmp_path)
    attempts = [{"finish_reason": "length", "usage": {"total_tokens": 42},
                 "request_max_tokens": 32, "content": "partial"}]

    def failed(*args, **kwargs):
        raise ModelError("model response truncated", transport_attempts=attempts)

    client.generate = failed
    with pytest.raises(ModelError, match="baseline fallback is disabled"):
        runner.run(request())
    failures = [event["data"] for event in runner.archive.events(
        "zero-llm-test", "zero-llm-round") if event["stage"] == "model_failed"]
    assert failures[0]["transport_attempts"] == attempts
