import math

import pytest

from ssa_harness.contracts import ForecastRequest
from ssa_harness.ensemble import BaselineFactory
from ssa_harness.ensemble.models import InvalidEnsembleMember


def numeric(factory, values, **fields):
    return {
        member.member_id: member
        for member in factory.build({
            "target_type": "continuous_normal", "history": values, **fields
        })
    }


def test_default_members_and_predictions_remain_compatible():
    members = numeric(BaselineFactory(), [10, 11, 12, 13])
    assert list(members) == ["persistence", "recent_mean"]
    assert members["persistence"].prediction == {"mean": 13, "sd": 0.25}
    assert members["recent_mean"].prediction == {"mean": 12, "sd": 0.25}


def test_opt_in_ewma_and_damped_local_trend():
    factory = BaselineFactory(
        numeric_members=["ewma", "local_trend"],
        ewma_alpha=0.5,
        trend_window=3,
        trend_damping=0.5,
        forecast_steps=2,
    )
    members = numeric(factory, [0, 2, 4, 6])
    assert list(members) == ["ewma", "local_trend"]
    assert members["ewma"].prediction["mean"] == 4.25
    # Slope 2, future increments are 2*0.5 and 2*0.5**2.
    assert members["local_trend"].prediction["mean"] == 7.5
    assert members["local_trend"].metadata["forecast_steps"] == 2


def test_local_trend_uses_only_configured_trailing_window():
    member = numeric(BaselineFactory(
        numeric_members=["local_trend"], trend_window=3, trend_damping=1
    ), [1000, 0, 1, 2])["local_trend"]
    assert member.prediction["mean"] == 3


def test_damped_trend_bounds_long_horizon_extrapolation():
    members = numeric(BaselineFactory(
        numeric_members=["local_trend"], trend_damping=0.5, forecast_steps=100
    ), [0, 2, 4])
    assert members["local_trend"].prediction["mean"] <= 6


def test_trend_and_residual_backtest_use_same_explicit_bounds():
    factory = BaselineFactory(
        numeric_members=["local_trend"], trend_damping=1,
        sd_method="rolling_residual", min_residuals=1,
    )
    member = numeric(factory, [97, 99, 100], minimum=0, maximum=100)["local_trend"]
    assert member.prediction["mean"] == 100
    assert member.metadata["unbounded_mean"] == 101.5
    assert member.metadata["mean_clipped_to_bounds"] is True
    # Historical prediction from [97, 99] is 101 and clips to actual target 100.
    assert member.prediction["sd"] == 0.25
    unbounded = numeric(factory, [97, 99, 100], unit="percent")["local_trend"]
    assert unbounded.prediction["mean"] == 101.5
    assert unbounded.prediction["sd"] == 1


def test_profile_trend_applies_explicit_bounds_to_every_cell():
    member = BaselineFactory(
        numeric_members=["local_trend"], trend_damping=1
    ).build({
        "target_type": "profile_energy", "cells": ["a", "b"],
        "minimum": 0, "maximum": 100,
        "history": [{"cells": {"a": 98, "b": 2}}, {"cells": {"a": 100, "b": 0}}],
    })[0]
    assert member.prediction["cells"]["a"]["mean"] == 100
    assert member.prediction["cells"]["b"]["mean"] == 0
    assert member.metadata["cells"]["a"]["mean_clipped_to_bounds"] is True
    assert member.metadata["cells"]["b"]["mean_clipped_to_bounds"] is True


def test_rolling_rmse_retains_constant_trend_bias():
    members = numeric(BaselineFactory(sd_method="rolling_residual"), [0, 1, 2, 3, 4, 5])
    persistence = members["persistence"]
    assert persistence.prediction["sd"] == 1
    assert persistence.metadata["sd_source"] == "rolling_residual_rmse"
    assert persistence.metadata["residual_count"] == 5
    # The recent mean strategy has a different, larger historical forecast error.
    assert members["recent_mean"].prediction["sd"] > persistence.prediction["sd"]


def test_residual_fit_never_sees_its_own_target():
    member = numeric(BaselineFactory(
        numeric_members=["persistence"], sd_method="rolling_residual"
    ), [0, 0, 0, 12])["persistence"]
    assert member.prediction["sd"] == pytest.approx(math.sqrt(48))
    assert member.metadata["residual_count"] == 3


def test_ewma_residuals_use_prefix_ewma_not_full_history_state():
    member = numeric(BaselineFactory(
        numeric_members=["ewma"], ewma_alpha=0.5, sd_method="rolling_residual"
    ), [0, 2, 4, 6])["ewma"]
    assert member.prediction["sd"] == pytest.approx(math.sqrt((4 + 9 + 12.25) / 3))


def test_h_step_residuals_match_h_step_means_and_do_not_scale_twice():
    members = numeric(BaselineFactory(
        numeric_members=["persistence", "local_trend"],
        sd_method="rolling_residual", forecast_steps=2, trend_damping=0.8
    ), [0, 1, 2, 3, 4, 5])
    assert members["persistence"].prediction["sd"] == 2
    assert members["persistence"].metadata["residual_count"] == 4
    trend = members["local_trend"]
    assert trend.prediction["mean"] == pytest.approx(6.44)
    assert trend.prediction["sd"] == pytest.approx(0.56)
    assert trend.metadata["residual_count"] == 3
    assert "sd_scaling" not in trend.metadata


def test_residual_window_keeps_recent_errors():
    member = numeric(BaselineFactory(
        numeric_members=["persistence"], sd_method="rolling_residual",
        residual_window=3, min_residuals=3
    ), [0, 100, 100, 100, 100])["persistence"]
    assert member.prediction["sd"] == 0.25
    assert member.metadata["residual_count"] == 3


def test_insufficient_residuals_report_explicit_fallback():
    member = numeric(BaselineFactory(
        numeric_members=["local_trend"], sd_method="rolling_residual", forecast_steps=2
    ), [12])["local_trend"]
    assert member.prediction["mean"] == 12
    assert member.prediction["sd"] == pytest.approx(0.25 * math.sqrt(2))
    assert member.metadata["residual_count"] == 0
    assert member.metadata["sd_source"] == "change_sd"
    assert "insufficient rolling residuals" in member.metadata["sd_fallback_reason"]
    assert "persistence" in member.metadata["mean_fallback"]


def test_profile_calibrates_each_cell_using_its_own_observations():
    request = {
        "target_type": "profile_energy",
        "cells": ["a", "b"],
        "history": [
            {"cells": {"a": 0, "b": 10}},
            {"cells": {"a": 1}},
            {"cells": {"a": 2}},
            {"cells": {"a": 3}},
            {"cells": {"a": 4, "b": 12}},
        ],
    }
    member = BaselineFactory(
        numeric_members=["persistence"], sd_method="rolling_residual"
    ).build(request)[0]
    assert member.prediction["cells"]["a"]["sd"] == 1
    assert member.metadata["cells"]["a"]["history_count"] == 5
    assert member.metadata["cells"]["a"]["residual_count"] == 4
    assert member.metadata["cells"]["b"]["history_count"] == 2
    assert member.metadata["cells"]["b"]["residual_count"] == 1
    assert member.metadata["cells"]["b"]["sd_source"] == "change_sd"
    assert "sd_fallback_reason" in member.metadata["cells"]["b"]


@pytest.mark.parametrize("legacy", [False, True])
def test_unavailable_future_observations_and_revisions_are_excluded(legacy):
    fields = {
        "round_id": "temporal-baseline",
        "question": "Future observation",
        "target_type": "continuous_normal",
        "unit": "points",
        "asof": "2025-01-05T12:00:00Z",
        "deadline": "2025-01-06T12:00:00Z",
        "history": [
            {"observed_at": f"2025-01-0{i}T00:00:00Z",
             "available_at": f"2025-01-0{i}T01:00:00Z", "value": i}
            for i in range(1, 6)
        ],
    }
    factory = BaselineFactory(
        numeric_members=list(BaselineFactory.NUMERIC_MEMBERS), sd_method="rolling_residual"
    )
    expected = factory.build(fields if legacy else ForecastRequest(**fields))
    fields["history"] += [
        {"observed_at": "2025-01-05T00:00:00Z",
         "available_at": "2025-01-06T00:00:00Z", "value": 1000},
        {"observed_at": "2025-01-06T00:00:00Z",
         "available_at": "2025-01-05T00:00:00Z", "value": 2000},
        {"observed_at": "2025-01-04T00:00:00Z",
         "available_at": "2025-01-04T00:00:00Z",
         "captured_at": "2025-01-06T00:00:00Z", "value": 3000},
    ]
    actual = factory.build(fields if legacy else ForecastRequest(**fields))
    assert actual == expected


def test_release_time_does_not_change_explicit_forecast_horizon():
    factory = BaselineFactory(numeric_members=["local_trend"], trend_damping=1)
    plain = numeric(factory, [1, 2, 3])
    dated = numeric(factory, [1, 2, 3], release_at="2099-12-31T00:00:00Z")
    assert plain == dated
    assert dated["local_trend"].prediction["mean"] == 4


@pytest.mark.parametrize("open_ranking", [False, True])
def test_numeric_member_selection_does_not_change_ranking_members(open_ranking):
    request = {
        "target_type": "ranking_list", "candidates": ["A", "B"],
        "history": [{"ranking": ["A", "B"]}],
    }
    if open_ranking:
        request["ranking_spec"] = {"length": 2}
    members = BaselineFactory(numeric_members=["ewma", "local_trend"]).build(request)
    assert [member.member_id for member in members] == ["persistence", "recent_mean"]


def test_identity_contains_all_constructor_settings_and_round_trips():
    configured = {
        "recent_window": 5, "min_sd": 0.5,
        "numeric_members": ["ewma", "local_trend"], "ewma_alpha": 0.4,
        "trend_window": 7, "trend_damping": 0.6, "sd_method": "rolling_residual",
        "residual_window": 9, "min_residuals": 4, "forecast_steps": 3,
    }
    assert BaselineFactory(**configured).identity() == configured
    assert BaselineFactory(**BaselineFactory().identity()).identity() == BaselineFactory().identity()


@pytest.mark.parametrize("config", [
    {"recent_window": True}, {"recent_window": 1.5}, {"recent_window": 0},
    {"min_sd": True}, {"min_sd": "1"}, {"min_sd": 0}, {"min_sd": float("inf")},
    {"numeric_members": []}, {"numeric_members": "ewma"},
    {"numeric_members": ["ewma", "ewma"]}, {"numeric_members": ["unknown"]},
    {"numeric_members": [["ewma"]]}, {"numeric_members": [1]},
    {"ewma_alpha": 0}, {"ewma_alpha": 1.1}, {"ewma_alpha": float("nan")},
    {"trend_window": 1}, {"trend_window": True},
    {"trend_damping": -0.1}, {"trend_damping": 1.1}, {"trend_damping": "0.5"},
    {"sd_method": "unknown"}, {"sd_method": None},
    {"residual_window": 0}, {"min_residuals": 0},
    {"residual_window": 2, "min_residuals": 3},
    {"forecast_steps": 0}, {"forecast_steps": 1.5}, {"forecast_steps": True},
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        BaselineFactory(**config)


def test_missing_numeric_and_profile_history_remains_an_error():
    with pytest.raises(InvalidEnsembleMember):
        numeric(BaselineFactory(numeric_members=["ewma"]), [])
    with pytest.raises(InvalidEnsembleMember):
        BaselineFactory().build({
            "target_type": "profile_energy", "cells": ["a", "b"],
            "history": [{"cells": {"a": 1, "b": True}}],
        })
