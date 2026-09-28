from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from ssa_harness.prompt._compat import field, string_list, target_type

from .models import ForecastMember, InvalidEnsembleMember


class BaselineFactory:
    """Build deterministic baselines from history available at the request cutoff.

    Numeric/profile members are opt-in; ranking keeps its two existing members.
    ``forecast_steps`` is an explicit count of observations, not a duration inferred
    from the request deadline or release time. Rolling residuals evaluate the same
    strategy at that horizon using only preceding observations. Their RMSE retains
    systematic forecast bias, unlike the standard deviation of residuals.
    """

    NUMERIC_MEMBERS = ("persistence", "recent_mean", "ewma", "local_trend")

    def __init__(
        self,
        *,
        recent_window: int = 3,
        min_sd: float = 0.25,
        numeric_members: tuple[str, ...] | list[str] = ("persistence", "recent_mean"),
        ewma_alpha: float = 0.35,
        trend_window: int = 6,
        trend_damping: float = 0.8,
        sd_method: str = "change_sd",
        residual_window: int = 20,
        min_residuals: int = 3,
        forecast_steps: int = 1,
    ) -> None:
        for name, value, lower in (
            ("recent_window", recent_window, 1),
            ("trend_window", trend_window, 2),
            ("residual_window", residual_window, 1),
            ("min_residuals", min_residuals, 1),
            ("forecast_steps", forecast_steps, 1),
        ):
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        for name, value in (
            ("min_sd", min_sd),
            ("ewma_alpha", ewma_alpha),
            ("trend_damping", trend_damping),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if min_sd <= 0:
            raise ValueError("min_sd must be positive")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0 <= trend_damping <= 1:
            raise ValueError("trend_damping must be in [0, 1]")
        if min_residuals > residual_window:
            raise ValueError("min_residuals cannot exceed residual_window")
        if sd_method not in ("change_sd", "rolling_residual"):
            raise ValueError("sd_method must be change_sd or rolling_residual")
        if (
            not isinstance(numeric_members, (tuple, list))
            or not numeric_members
            or any(
                not isinstance(name, str) or name not in self.NUMERIC_MEMBERS
                for name in numeric_members
            )
            or len(set(numeric_members)) != len(numeric_members)
        ):
            raise ValueError("numeric_members must contain unique supported strategies")
        self.recent_window = recent_window
        self.min_sd = float(min_sd)
        self.numeric_members = tuple(numeric_members)
        self.ewma_alpha = float(ewma_alpha)
        self.trend_window = trend_window
        self.trend_damping = float(trend_damping)
        self.sd_method = sd_method
        self.residual_window = residual_window
        self.min_residuals = min_residuals
        self.forecast_steps = forecast_steps

    def identity(self) -> dict[str, Any]:
        """Return every constructor setting for cache/run identity and audit."""
        return {
            "recent_window": self.recent_window,
            "min_sd": self.min_sd,
            "numeric_members": list(self.numeric_members),
            "ewma_alpha": self.ewma_alpha,
            "trend_window": self.trend_window,
            "trend_damping": self.trend_damping,
            "sd_method": self.sd_method,
            "residual_window": self.residual_window,
            "min_residuals": self.min_residuals,
            "forecast_steps": self.forecast_steps,
        }

    def build(self, request: Any) -> list[ForecastMember]:
        kind = target_type(request)
        history = field(request, "history", [])
        if hasattr(request, "request_hash"):
            from ..history import normalize_history

            history = normalize_history(request).rows
        elif field(request, "asof", None):
            # Legacy dictionary inputs with a cutoff must provide availability.
            from ..contracts import ForecastRequest
            from ..history import normalize_history

            values = dict(request)
            values.setdefault("question", "legacy baseline request")
            values.setdefault("unit", "points")
            values.setdefault("round_id", "legacy-baseline")
            if "deadline" not in values:
                from datetime import timedelta
                from ..retrieval.utils import iso, parse_instant

                values["deadline"] = iso(
                    parse_instant(values["asof"]) + timedelta(days=1)
                )
            allowed = ForecastRequest.__dataclass_fields__
            history = normalize_history(
                ForecastRequest(**{k: v for k, v in values.items() if k in allowed})
            ).rows
        if kind == "continuous_normal":
            return self._numeric(history, self._bounds(request))
        if kind == "profile_energy":
            return self._profile(
                history, string_list(field(request, "cells"), "cells"), self._bounds(request)
            )
        if kind == "ranking_list":
            spec = field(request, "ranking_spec", {})
            if spec and not spec.get("items"):
                return self._open_ranking(history, spec)
            return self._ranking(
                history,
                string_list(field(request, "candidates"), "candidates"),
            )
        raise InvalidEnsembleMember(f"unsupported target type: {kind}")

    @staticmethod
    def _bounds(request: Any) -> tuple[float | None, float | None]:
        minimum, maximum = field(request, "minimum", None), field(request, "maximum", None)
        for bound in (minimum, maximum):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
            ):
                raise InvalidEnsembleMember("baseline bounds must be finite numbers")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise InvalidEnsembleMember("baseline minimum exceeds maximum")
        return minimum, maximum

    @staticmethod
    def _bounded_mean(mean: float, bounds: tuple[float | None, float | None]) -> float:
        minimum, maximum = bounds
        if minimum is not None:
            mean = max(minimum, mean)
        if maximum is not None:
            mean = min(maximum, mean)
        return mean

    def _numeric(self, history: Any, bounds=(None, None)) -> list[ForecastMember]:
        values = self._numeric_values(history)
        if not values:
            raise InvalidEnsembleMember("numeric baseline requires history")
        members = []
        for name in self.numeric_members:
            prediction, metadata = self._forecast(values, name, bounds)
            members.append(ForecastMember(name, name, prediction, metadata=metadata))
        return members

    def _profile(
        self, history: Any, cells: list[str], bounds=(None, None)
    ) -> list[ForecastMember]:
        per_cell = {cell: [] for cell in cells}
        for row in history or []:
            values = self._row_value(row)
            if isinstance(values, Mapping) and "cells" in values:
                values = values["cells"]
            if not isinstance(values, Mapping):
                continue
            for cell in cells:
                if cell in values:
                    cell_value = values[cell]
                    if isinstance(cell_value, Mapping):
                        cell_value = cell_value.get("mean", cell_value.get("value"))
                    if isinstance(cell_value, bool):
                        continue
                    try:
                        number = float(cell_value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        per_cell[cell].append(number)
        missing = [cell for cell, values in per_cell.items() if not values]
        if missing:
            raise InvalidEnsembleMember(f"profile history missing cells: {missing}")

        members = []
        for name in self.numeric_members:
            predictions, details = {}, {}
            for cell, values in per_cell.items():
                predictions[cell], details[cell] = self._forecast(values, name, bounds)
            members.append(
                ForecastMember(
                    name,
                    name,
                    {"cells": predictions},
                    metadata={
                        "forecast_steps": self.forecast_steps,
                        "sd_method": self.sd_method,
                        "cells": details,
                    },
                )
            )
        return members

    def _predict_mean(self, values: list[float], strategy: str) -> float:
        if strategy == "persistence":
            return values[-1]
        if strategy == "recent_mean":
            return statistics.fmean(values[-self.recent_window :])
        if strategy == "ewma":
            level = values[0]
            for value in values[1:]:
                level = self.ewma_alpha * value + (1 - self.ewma_alpha) * level
            return level
        # Least-squares slope on a trailing window, anchored at the last value.
        # Damping each future increment bounds long-horizon extrapolation when
        # trend_damping < 1, while 0 explicitly disables the trend adjustment.
        recent = values[-self.trend_window :]
        if len(recent) < 2:
            return values[-1]
        center = (len(recent) - 1) / 2
        mean = statistics.fmean(recent)
        slope = sum((i - center) * (value - mean) for i, value in enumerate(recent))
        slope /= sum((i - center) ** 2 for i in range(len(recent)))
        damping = self.trend_damping
        if damping == 1:
            horizon = self.forecast_steps
        else:
            horizon = damping * (1 - damping ** self.forecast_steps) / (1 - damping)
        return values[-1] + slope * horizon

    def _forecast(self, values: list[float], strategy: str, bounds=(None, None)):
        metadata = {
            "mean_strategy": strategy,
            "history_count": len(values),
            "forecast_steps": self.forecast_steps,
            "sd_method": self.sd_method,
            "residual_count": 0,
        }
        if strategy == "local_trend" and len(values) < 2:
            metadata["mean_fallback"] = "persistence: fewer than two observations"
        # The legacy change scale remains the default, preserving one-step
        # behavior. Its multi-step fallback assumes independent increments.
        sd = self._change_sd(values) * math.sqrt(self.forecast_steps)
        metadata["sd_source"] = "change_sd"
        if self.forecast_steps != 1:
            metadata["sd_scaling"] = "sqrt_forecast_steps"
        if self.sd_method == "rolling_residual":
            # origin is the number of observations available for a forecast;
            # its h-step target is at origin+h-1. No target enters its own fit.
            min_prefix = 2 if strategy == "local_trend" else 1
            stop = len(values) - self.forecast_steps + 1
            start = max(min_prefix, stop - self.residual_window)
            residuals = [
                values[origin + self.forecast_steps - 1]
                - self._bounded_mean(self._predict_mean(values[:origin], strategy), bounds)
                for origin in range(start, stop)
            ]
            metadata["residual_count"] = len(residuals)
            if len(residuals) >= self.min_residuals:
                sd = max(
                    self.min_sd,
                    math.sqrt(statistics.fmean(error * error for error in residuals)),
                )
                metadata["sd_source"] = "rolling_residual_rmse"
                metadata.pop("sd_scaling", None)
            else:
                metadata["sd_fallback_reason"] = (
                    f"insufficient rolling residuals: {len(residuals)} < {self.min_residuals}"
                )
        unbounded_mean = self._predict_mean(values, strategy)
        mean = self._bounded_mean(unbounded_mean, bounds)
        if mean != unbounded_mean:
            metadata["mean_clipped_to_bounds"] = True
            metadata["unbounded_mean"] = unbounded_mean
        return {"mean": mean, "sd": sd}, metadata

    def _ranking(self, history: Any, candidates: list[str]) -> list[ForecastMember]:
        rankings: list[list[str]] = []
        for row in history or []:
            value = self._row_value(row)
            if isinstance(value, Mapping):
                value = value.get("ranking")
            if (
                isinstance(value, (list, tuple))
                and len(value) == len(candidates)
                and len(set(value)) == len(value)
                and set(value) == set(candidates)
            ):
                rankings.append([str(item) for item in value])
        if not rankings:
            raise InvalidEnsembleMember(
                "ranking baseline requires a complete historical ranking"
            )

        latest = rankings[-1]
        persistence_scores = self._borda(latest)
        recent = rankings[-self.recent_window :]
        recent_scores = {
            item: statistics.fmean(self._borda(ranking)[item] for ranking in recent)
            for item in candidates
        }
        order = {item: index for index, item in enumerate(candidates)}
        recent_ranking = sorted(
            candidates,
            key=lambda item: (-recent_scores[item], order[item]),
        )
        return [
            ForecastMember(
                "persistence",
                "persistence",
                {"scores": persistence_scores, "ranking": latest},
            ),
            ForecastMember(
                "recent_mean",
                "recent_mean",
                {"scores": recent_scores, "ranking": recent_ranking},
            ),
        ]

    def _numeric_values(self, history: Any) -> list[float]:
        output: list[float] = []
        for row in history or []:
            value = self._row_value(row)
            if isinstance(value, Mapping):
                value = value.get("mean", value.get("value"))
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output.append(number)
        return output

    @staticmethod
    def _row_value(row: Any) -> Any:
        if isinstance(row, Mapping):
            for name in (
                "value",
                "truth",
                "result",
                "prediction",
                "ranking",
                "items",
                "profile",
                "cells",
            ):
                if name in row:
                    return row[name]
        return row

    def _change_sd(self, values: list[float]) -> float:
        deltas = [b - a for a, b in zip(values, values[1:])]
        if len(deltas) < 2:
            return self.min_sd
        return max(self.min_sd, statistics.pstdev(deltas))

    @staticmethod
    def _borda(ranking: list[str]) -> dict[str, float]:
        size = len(ranking)
        return {item: float(size - index) for index, item in enumerate(ranking)}

    def _open_ranking(self, history, spec):
        from ..ranking import normalize_title

        rankings = []
        for row in history:
            values = self._row_value(row)
            if not isinstance(values, (list, tuple)):
                continue
            ranking = [normalize_title(x, spec) for x in values]
            if len(ranking) >= spec["length"] and len(set(ranking)) == len(ranking):
                rankings.append(ranking[: spec["length"]])
        if not rankings:
            raise InvalidEnsembleMember("no complete historical open list")
        scores = {}
        for ranking in rankings[-self.recent_window :]:
            for i, item in enumerate(ranking):
                scores[item] = scores.get(item, 0) + 1 / (i + 1)
        latest = rankings[-1]
        ties = {item: i for i, item in enumerate(latest)}
        recent = sorted(scores, key=lambda x: (-scores[x], ties.get(x, len(ties)), x))[
            : spec["length"]
        ]
        return [
            ForecastMember("persistence", "persistence", {"ranking": latest}),
            ForecastMember("recent_mean", "recent_mean", {"ranking": recent}),
        ]
