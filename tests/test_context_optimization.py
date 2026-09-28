"""Regression coverage for model-visible temporal and numerical information."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from ssa_harness.context import ContextBuilder, ContextConfig
from ssa_harness.contracts import ForecastRequest
from ssa_harness.prompt import ForecastPromptCompiler


def request(**changes):
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [{"date": (start + timedelta(days=i)).isoformat(),
             "available_at": (start + timedelta(days=i, hours=1)).isoformat(),
             "value": float(i)} for i in range(30)]
    return ForecastRequest(**dict({
        "round_id": "context-regression", "question": "Approval indicator",
        "unit": "percent", "target_type": "continuous_normal",
        "asof": "2026-08-01T00:00:00Z", "deadline": "2026-08-02T00:00:00Z",
        "release_at": "2026-08-04T00:00:00Z", "history": rows,
    }, **changes))


def section(context, name):
    text = context.rendered_text.split(name + "\n", 1)[1].split("\n\n", 1)[0]
    return json.loads(text)


def test_release_is_visible_without_inventing_measurement_window():
    req = request()
    context = ContextBuilder().build(req)
    compiled = ForecastPromptCompiler().compile(req, context)
    assert "release scheduled at 2026-08-04" in compiled.user_message
    assert "measurement window is not supplied" in compiled.user_message
    assert section(context, "TIME INFORMATION")["days_until_scheduled_release"] == 3
    assert section(context, "TASK")["target_end"] is None
    assert "Forecast target time/window:\n2026-08-02" not in compiled.user_message


def test_explicit_window_is_distinct_from_release_and_deadline():
    req = request(target_start="2026-08-02", target_end="2026-08-03")
    prompt = ForecastPromptCompiler().compile(req, ContextBuilder().build(req)).user_message
    assert "2026-08-02 through 2026-08-03" in prompt
    assert "Scheduled publication time (not the measurement window):\n2026-08-04" in prompt


def test_unknown_time_stays_unknown():
    req = request(release_at=None)
    context = ContextBuilder().build(req)
    prompt = ForecastPromptCompiler().compile(req, context).user_message
    assert "Measurement window not supplied" in prompt
    assert section(context, "TIME INFORMATION")["days_until_scheduled_release"] is None


def test_raw_history_window_is_independent_of_summary_window_and_filters_future():
    req = request()
    future = {"date": "2026-07-31", "available_at": "2026-08-03T00:00:00Z", "value": 999}
    req = replace(req, history=(*req.history, future))
    ctx = ContextBuilder(ContextConfig(long_window=5, recent_window=2)).build(req)
    rows = section(ctx, "RECENT NUMERICAL HISTORY")
    assert len(rows) == 24
    assert rows[0]["value"] == 6
    assert rows[-1]["value"] == 29
    assert ctx.history_summary["summary_window_points"] == 5
    assert ctx.history_summary["long_term_mean"] == 27
    assert "999" not in ctx.rendered_text
    ctx.verify()


def test_profile_matrix_retains_twelve_points_and_missing_values_under_budget():
    cells = tuple(f"approval_age_education_subgroup_{i}" for i in range(16))
    history = []
    for row in request().history[-12:]:
        vals = {c: row["value"] + i for i, c in enumerate(cells)}
        if row == request().history[-1]:
            vals.pop(cells[-1])
        history.append({**row, "value": vals})
    req = request(target_type="profile_energy", cells=cells, history=history)
    ctx = ContextBuilder().build(req)
    matrix = section(ctx, "RECENT NUMERICAL HISTORY")
    assert len(matrix["dates"]) == 12
    assert matrix["displayed_points_by_cell"][cells[0]] == 12
    assert matrix["values_by_cell"][cells[-1]][-1] is None
    assert matrix["values_by_cell"][cells[0]][0] == 18
    assert len(ctx.rendered_text) <= 14000
    ctx.verify()


def test_ranking_preserves_continuous_signals_and_budget_is_not_silent_truncation():
    row = {"date": "2026-07-20", "available_at": "2026-07-21T00:00:00Z",
           "value": ["A", "B"], "views": {"A": 120, "B": 11}, "index": {"A": 100, "B": 9}}
    req = request(target_type="ranking_list", ranking_spec={"length": 2}, history=[row])
    ctx = ContextBuilder().build(req)
    rendered = section(ctx, "RECENT NUMERICAL HISTORY")[0]
    assert rendered["views"] == row["views"]
    assert rendered["index"] == row["index"]
    with pytest.raises(ValueError, match="budget"):
        ContextBuilder(ContextConfig(max_chars=50)).build(req)


def test_profile_equivalent_timestamps_share_one_deterministic_column():
    req = request(target_type="profile_energy", cells=("a", "b"), history=[
        {"date": "2026-07-01T12:00:00Z", "available_at": "2026-07-02T00:00:00Z", "value": {"a": 1}},
        {"date": "2026-07-01T12:00:00+00:00", "available_at": "2026-07-02T00:00:00Z", "value": {"b": 2}},
    ])
    matrix = section(ContextBuilder().build(req), "RECENT NUMERICAL HISTORY")
    assert len(matrix["dates"]) == 1
    assert matrix["values_by_cell"] == {"a": [1.0], "b": [2.0]}


def test_profile_fractional_seconds_sort_by_instant():
    req = request(target_type="profile_energy", cells=("a", "b"), history=[
        {"date": "2026-07-01T12:00:00.500000Z", "available_at": "2026-07-02T00:00:00Z", "value": {"a": 2, "b": 4}},
        {"date": "2026-07-01T12:00:00Z", "available_at": "2026-07-02T00:00:00Z", "value": {"a": 1, "b": 3}},
    ])
    matrix = section(ContextBuilder().build(req), "RECENT NUMERICAL HISTORY")
    assert matrix["values_by_cell"] == {"a": [1.0, 2.0], "b": [3.0, 4.0]}
