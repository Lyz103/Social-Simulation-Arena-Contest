from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from ._compat import field, string_list, target_type
from .models import InvalidModelOutput


class PredictionValidator:
    """Parse and validate all three SSA prediction formats."""

    def parse(self, response: str | Mapping[str, Any], request: Any) -> dict[str, Any]:
        obj = self._object(response)
        kind = target_type(request)
        if kind == "continuous_normal":
            return self._normal(obj, request)
        if kind == "profile_energy":
            return self._profile(obj, request)
        if kind == "ranking_list":
            return self._ranking(obj, request)
        raise InvalidModelOutput(f"unsupported target type: {kind}")

    def _object(self, response: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(response, Mapping):
            return dict(response)
        if not isinstance(response, str):
            raise InvalidModelOutput("response must be JSON text or a mapping")
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:].lstrip("\r\n")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutput(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise InvalidModelOutput("top-level response must be an object")
        return obj

    def _normal(self, obj: Mapping[str, Any], request: Any) -> dict[str, float]:
        if not isinstance(obj, Mapping):
            raise InvalidModelOutput("normal prediction must be an object")
        if set(obj) != {"mean", "sd"}:
            raise InvalidModelOutput("normal prediction requires exactly mean and sd")
        mean = self._finite(obj["mean"], "mean")
        sd = self._finite(obj["sd"], "sd")
        minimum = field(request, "minimum", None)
        maximum = field(request, "maximum", None)
        if minimum is not None and mean < float(minimum):
            raise InvalidModelOutput("mean is below the valid range")
        if maximum is not None and mean > float(maximum):
            raise InvalidModelOutput("mean is above the valid range")
        if sd <= 0 or sd > 1e9:
            raise InvalidModelOutput("sd must be positive and at most 1e9")
        return {"mean": mean, "sd": sd}

    def _profile(self, obj: Mapping[str, Any], request: Any) -> dict[str, Any]:
        if set(obj) != {"cells"} or not isinstance(obj.get("cells"), Mapping):
            raise InvalidModelOutput(
                "profile prediction requires exactly a cells object"
            )
        required = string_list(field(request, "cells"), "cells")
        cells = obj["cells"]
        if set(cells) != set(required):
            missing = sorted(set(required) - set(cells))
            extra = sorted(set(cells) - set(required))
            raise InvalidModelOutput(
                f"profile cells mismatch; missing={missing}, extra={extra}"
            )
        return {
            "cells": {cell: self._normal(cells[cell], request) for cell in required}
        }

    def _ranking(self, obj: Mapping[str, Any], request: Any) -> dict[str, Any]:
        spec = field(request, "ranking_spec", {})
        if spec and not spec.get("items"):
            from ..ranking import normalize_title

            if set(obj) != {"ranking"} or not isinstance(obj["ranking"], list):
                raise InvalidModelOutput(
                    "open ranking requires exactly a ranking array"
                )
            try:
                ranking = [normalize_title(item, spec) for item in obj["ranking"]]
            except ValueError as exc:
                raise InvalidModelOutput(str(exc)) from exc
            if len(ranking) != spec["length"] or len(set(ranking)) != len(ranking):
                raise InvalidModelOutput("open ranking length/uniqueness mismatch")
            return {"ranking": ranking}
        if set(obj) != {"scores", "ranking"}:
            raise InvalidModelOutput("ranking prediction requires scores and ranking")
        if not isinstance(obj["scores"], Mapping) or not isinstance(
            obj["ranking"], list
        ):
            raise InvalidModelOutput(
                "scores must be an object and ranking must be a list"
            )
        candidates = string_list(field(request, "candidates"), "candidates")
        if set(obj["scores"]) != set(candidates):
            raise InvalidModelOutput("score candidates do not match request candidates")
        if any(not isinstance(item, str) for item in obj["ranking"]):
            raise InvalidModelOutput("ranking items must be strings")
        ranking = obj["ranking"]
        if len(ranking) != len(candidates) or set(ranking) != set(candidates):
            raise InvalidModelOutput(
                "ranking must contain every candidate exactly once"
            )
        scores = {
            candidate: self._finite(obj["scores"][candidate], f"score[{candidate}]")
            for candidate in candidates
        }
        order = {candidate: index for index, candidate in enumerate(candidates)}
        rebuilt = sorted(candidates, key=lambda item: (-scores[item], order[item]))
        return {"scores": scores, "ranking": rebuilt}

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidModelOutput(f"{name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidModelOutput(f"{name} must be numeric") from exc
        if not math.isfinite(number):
            raise InvalidModelOutput(f"{name} must be finite")
        return number
