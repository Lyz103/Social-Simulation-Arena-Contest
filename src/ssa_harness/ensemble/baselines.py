from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from ssa_harness.prompt._compat import field, string_list, target_type

from .models import ForecastMember, InvalidEnsembleMember


class BaselineFactory:
    """Build persistence and recent-mean baselines from common history formats."""

    def __init__(self, *, recent_window: int = 3, min_sd: float = 0.25) -> None:
        if (
            type(recent_window) is not int
            or recent_window < 1
            or not math.isfinite(min_sd)
            or min_sd <= 0
        ):
            raise ValueError("recent_window and min_sd must be positive")
        self.recent_window = recent_window
        self.min_sd = float(min_sd)

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
            return self._numeric(history)
        if kind == "profile_energy":
            return self._profile(history, string_list(field(request, "cells"), "cells"))
        if kind == "ranking_list":
            spec = field(request, "ranking_spec", {})
            if spec and not spec.get("items"):
                return self._open_ranking(history, spec)
            return self._ranking(
                history,
                string_list(field(request, "candidates"), "candidates"),
            )
        raise InvalidEnsembleMember(f"unsupported target type: {kind}")

    def _numeric(self, history: Any) -> list[ForecastMember]:
        values = self._numeric_values(history)
        if not values:
            raise InvalidEnsembleMember("numeric baseline requires history")
        sd = self._change_sd(values)
        recent = values[-self.recent_window :]
        return [
            ForecastMember(
                "persistence",
                "persistence",
                {"mean": values[-1], "sd": sd},
            ),
            ForecastMember(
                "recent_mean",
                "recent_mean",
                {"mean": statistics.fmean(recent), "sd": sd},
            ),
        ]

    def _profile(self, history: Any, cells: list[str]) -> list[ForecastMember]:
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
                    try:
                        number = float(cell_value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        per_cell[cell].append(number)
        missing = [cell for cell, values in per_cell.items() if not values]
        if missing:
            raise InvalidEnsembleMember(f"profile history missing cells: {missing}")

        persistence: dict[str, dict[str, float]] = {}
        recent_mean: dict[str, dict[str, float]] = {}
        for cell, values in per_cell.items():
            sd = self._change_sd(values)
            persistence[cell] = {"mean": values[-1], "sd": sd}
            recent_mean[cell] = {
                "mean": statistics.fmean(values[-self.recent_window :]),
                "sd": sd,
            }
        return [
            ForecastMember("persistence", "persistence", {"cells": persistence}),
            ForecastMember("recent_mean", "recent_mean", {"cells": recent_mean}),
        ]

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
