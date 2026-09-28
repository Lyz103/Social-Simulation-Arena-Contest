from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ssa_harness.prompt._compat import field, string_list, target_type
from ssa_harness.prompt.validators import PredictionValidator
from ssa_harness.prompt.models import InvalidModelOutput

from .baselines import BaselineFactory
from .models import (
    FinalPrediction,
    ForecastMember,
    InvalidEnsembleMember,
    InvalidEnsembleWeights,
)


DEFAULT_WEIGHTS = {
    "llm": 0.60,
    "persistence": 0.25,
    "recent_mean": 0.15,
}

POLICY_MATCH_FIELDS = frozenset(
    {"target_type", "source_name", "metric", "population", "unit"}
)


class EnsembleEngine:
    """Combine validated LLM forecasts and statistical baselines."""

    VERSION = "ensemble-v3"

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        min_sd: float = 0.05,
        max_sd: float | None = None,
        baseline_factory: BaselineFactory | None = None,
        failure_policy: str = "strict",
        task_policies: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if (
            not math.isfinite(min_sd)
            or min_sd <= 0
            or (max_sd is not None and (not math.isfinite(max_sd) or max_sd < min_sd))
        ):
            raise ValueError("invalid sd limits")
        if failure_policy not in ("strict", "drop_invalid"):
            raise ValueError("unknown failure policy")
        self.failure_policy = failure_policy
        self.configured_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.task_policies = self._validate_policies(task_policies)
        self.min_sd = float(min_sd)
        self.max_sd = float(max_sd) if max_sd is not None else None
        self.baseline_factory = baseline_factory or BaselineFactory()
        self.validator = PredictionValidator()

    @staticmethod
    def _validate_policies(policies):
        if policies is None:
            return []
        if not isinstance(policies, (list, tuple)):
            raise ValueError("task_policies must be a list")
        result = []
        for policy in policies:
            if not isinstance(policy, Mapping) or set(policy) - {
                "match", "weights", "sd_scale"
            }:
                raise ValueError("invalid task policy fields")
            match = policy.get("match")
            if (
                not isinstance(match, Mapping)
                or set(match) - POLICY_MATCH_FIELDS
                or any(not isinstance(value, str) for value in match.values())
            ):
                raise ValueError("task policy match must use allowed exact string fields")
            scale = policy.get("sd_scale", 1.0)
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError("task policy sd_scale must be finite and positive")
            normalized = {"match": dict(match), "sd_scale": float(scale)}
            if "weights" in policy:
                weights = policy["weights"]
                if (
                    not isinstance(weights, Mapping)
                    or not weights
                    or any(not isinstance(key, str) or not key for key in weights)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        for value in weights.values()
                    )
                    or not any(value > 0 for value in weights.values())
                ):
                    raise InvalidEnsembleWeights("invalid task policy weights")
                normalized["weights"] = {
                    key: float(value) for key, value in weights.items()
                }
            result.append(normalized)
        return result

    def resolve_policy(self, request: Any) -> dict[str, Any]:
        """Select exact task metadata matches; ties are errors, never list-order wins."""
        matches = [
            (index, policy)
            for index, policy in enumerate(self.task_policies)
            if all(
                (target_type(request) if key == "target_type" else field(request, key, ""))
                == value
                for key, value in policy["match"].items()
            )
        ]
        if matches:
            specificity = max(len(policy["match"]) for _, policy in matches)
            selected = [
                item for item in matches if len(item[1]["match"]) == specificity
            ]
            if len(selected) != 1:
                raise ValueError("ambiguous task policies at the same specificity")
            index, policy = selected[0]
        else:
            index, policy = None, {}
        return {
            "index": index,
            "match": dict(policy.get("match", {})),
            "configured_weights": dict(policy.get("weights", self.configured_weights)),
            "policy_sd_scale": policy.get("sd_scale", 1.0),
        }

    def combine(
        self,
        request: Any,
        llm_predictions: Sequence[Any],
        baselines: Sequence[ForecastMember | Mapping[str, Any]] | None = None,
        *,
        sd_scale: float = 1.0,
    ) -> FinalPrediction:
        policy = self.resolve_policy(request)
        if (
            isinstance(sd_scale, bool)
            or not isinstance(sd_scale, (int, float))
            or not math.isfinite(sd_scale)
            or sd_scale <= 0
        ):
            raise ValueError("sd_scale must be finite and positive")
        effective_scale = policy["policy_sd_scale"] * sd_scale
        if not math.isfinite(effective_scale) or effective_scale <= 0:
            raise ValueError("combined sd_scale must be finite and positive")
        members, failures = [], []
        for index, item in enumerate(llm_predictions):
            try:
                member = self._llm_members(request, [item])[0]
                explicit_id = self._optional(
                    item, "member_id", self._optional(item, "model_id", None)
                )
                if explicit_id is None and not isinstance(item, ForecastMember):
                    member = ForecastMember(
                        f"llm-{index+1}",
                        member.member_type,
                        member.prediction,
                        member.weight,
                        member.metadata,
                    )
                members.append(member)
            except InvalidModelOutput as exc:
                if self.failure_policy == "strict":
                    raise
                failures.append(f"LLM member {index} excluded: {exc}")
        try:
            base_input = (
                baselines
                if baselines is not None
                else self.baseline_factory.build(request)
            )
        except InvalidEnsembleMember as exc:
            if self.failure_policy == "strict":
                raise
            base_input = []
            failures.append(f"baselines unavailable: {exc}")
        for item in base_input:
            try:
                member = self._coerce_member(item)
                self.validator.parse(member.prediction, request)
                members.append(member)
            except (InvalidModelOutput, InvalidEnsembleMember) as exc:
                if self.failure_policy == "strict":
                    raise
                failures.append(f"baseline excluded: {exc}")
        if not members:
            raise InvalidEnsembleMember("ensemble has no members")

        weights = self._weights(members, policy["configured_weights"])
        active_members = [member for member in members if weights[member.member_id] > 0]
        kind = target_type(request)
        if kind == "continuous_normal":
            prediction, warnings = self._numeric(request, active_members, weights)
        elif kind == "profile_energy":
            prediction, warnings = self._profile(request, active_members, weights)
        elif kind == "ranking_list":
            prediction, warnings = self._ranking(request, active_members, weights)
        else:
            raise InvalidEnsembleMember(f"unsupported target type: {kind}")

        if kind != "ranking_list" and effective_scale != 1:
            normals = (
                prediction["cells"].values() if kind == "profile_energy" else [prediction]
            )
            for normal in normals:
                normal["sd"] *= effective_scale
            warnings.append("sd scale applied: " + str(effective_scale))
        policy.update(
            global_sd_scale=float(sd_scale),
            effective_sd_scale=effective_scale if kind != "ranking_list" else 1.0,
            normalized_weights=dict(weights),
        )

        # Final validation also canonicalizes numeric and ranking output.
        prediction = self.validator.parse(prediction, request)
        warnings = failures + warnings
        return FinalPrediction(
            round_id=str(field(request, "round_id")),
            target_type=kind,
            prediction=prediction,
            members=members,
            weights=weights,
            ensemble_version=self.VERSION,
            warnings=warnings,
            degraded=bool(failures)
            or (
                policy["configured_weights"].get("llm", 0) > 0
                and not any(
                    member.member_type == "llm" and weights[member.member_id] > 0
                    for member in members
                )
            ),
            policy=policy,
        )

    def _llm_members(
        self, request: Any, predictions: Sequence[Any]
    ) -> list[ForecastMember]:
        output: list[ForecastMember] = []
        for index, item in enumerate(predictions):
            if isinstance(item, ForecastMember):
                prediction = self.validator.parse(item.prediction, request)
                output.append(
                    ForecastMember(
                        item.member_id,
                        item.member_type,
                        prediction,
                        item.weight,
                        dict(item.metadata),
                    )
                )
                continue

            prediction = self._prediction_payload(item)
            prediction = self.validator.parse(prediction, request)
            member_id = str(
                self._optional(
                    item,
                    "member_id",
                    self._optional(item, "model_id", f"llm-{index + 1}"),
                )
            )
            output.append(ForecastMember(member_id, "llm", prediction))
        return output

    @staticmethod
    def _prediction_payload(item: Any) -> Any:
        if isinstance(item, Mapping):
            if "prediction" in item:
                return item["prediction"]
            if "parsed" in item:
                return item["parsed"]
            return item
        if hasattr(item, "prediction"):
            return item.prediction
        if hasattr(item, "parsed"):
            return item.parsed
        return item

    @staticmethod
    def _optional(obj: Any, name: str, default: Any) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _coerce_member(item: ForecastMember | Mapping[str, Any]) -> ForecastMember:
        if isinstance(item, ForecastMember):
            return item
        if not isinstance(item, Mapping):
            raise InvalidEnsembleMember(
                "baseline member must be ForecastMember or mapping"
            )
        try:
            return ForecastMember(
                member_id=str(item["member_id"]),
                member_type=str(item["member_type"]),
                prediction=dict(item["prediction"]),
                weight=(
                    float(item["weight"]) if item.get("weight") is not None else None
                ),
                metadata=dict(item.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidEnsembleMember("invalid baseline member mapping") from exc

    def _weights(
        self, members: list[ForecastMember], configured_weights: Mapping[str, float]
    ) -> dict[str, float]:
        raw: dict[str, float] = {}
        type_counts: dict[str, int] = {}
        for member in members:
            type_counts[member.member_type] = type_counts.get(member.member_type, 0) + 1
        for member in members:
            if member.member_id in raw:
                raise InvalidEnsembleMember(f"duplicate member_id: {member.member_id}")
            configured = configured_weights.get(member.member_type, 0.0)
            value = (
                member.weight
                if member.weight is not None
                else configured / type_counts[member.member_type]
            )
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise InvalidEnsembleWeights(f"invalid weight for {member.member_id}")
            raw[member.member_id] = value
        scale = max(raw.values(), default=0)
        if scale <= 0:
            raise InvalidEnsembleWeights("active member weights sum to zero")
        scaled = {name: value / scale for name, value in raw.items()}
        total = math.fsum(scaled.values())
        return {name: value / total for name, value in scaled.items()}

    def _numeric(
        self,
        request: Any,
        members: list[ForecastMember],
        weights: Mapping[str, float],
    ) -> tuple[dict[str, float], list[str]]:
        normals = [
            self.validator.parse(member.prediction, request) for member in members
        ]
        result, warnings = self._mix_normals(members, normals, weights)
        return self._clip_normal(request, result, warnings), warnings

    def _profile(
        self,
        request: Any,
        members: list[ForecastMember],
        weights: Mapping[str, float],
    ) -> tuple[dict[str, Any], list[str]]:
        validated = [
            self.validator.parse(member.prediction, request) for member in members
        ]
        warnings: list[str] = []
        cells: dict[str, dict[str, float]] = {}
        for cell in string_list(field(request, "cells"), "cells"):
            normals = [item["cells"][cell] for item in validated]
            mixed, cell_warnings = self._mix_normals(members, normals, weights)
            cells[cell] = self._clip_normal(request, mixed, cell_warnings, label=cell)
            warnings.extend(cell_warnings)
        return {"cells": cells}, warnings

    def _ranking(
        self,
        request: Any,
        members: list[ForecastMember],
        weights: Mapping[str, float],
    ) -> tuple[dict[str, Any], list[str]]:
        spec = field(request, "ranking_spec", {})
        if spec and not spec.get("items"):
            validated = [
                self.validator.parse(m.prediction, request)["ranking"] for m in members
            ]
            # Rank-based scores make lists with different candidate sets comparable.
            scores = {}
            for member, ranking in zip(members, validated):
                for rank, item in enumerate(ranking):
                    scores[item] = scores.get(item, 0) + weights[member.member_id] / (
                        rank + 1
                    )
            last = next(
                (
                    r
                    for m, r in zip(members, validated)
                    if m.member_type == "persistence"
                ),
                [],
            )
            tie = {item: i for i, item in enumerate(last)}
            ranking = sorted(
                scores, key=lambda x: (-scores[x], tie.get(x, len(tie)), x)
            )[: spec["length"]]
            return {"ranking": ranking}, []
        candidates = string_list(field(request, "candidates"), "candidates")
        validated = [
            self.validator.parse(member.prediction, request) for member in members
        ]
        normalized: list[dict[str, float]] = []
        for item in validated:
            scores = item["scores"]
            low, high = min(scores.values()), max(scores.values())
            if high == low:
                normalized.append({candidate: 0.5 for candidate in candidates})
            else:
                scale = max(abs(low), abs(high))
                scaled_low, scaled_high = low / scale, high / scale
                normalized.append(
                    {
                        candidate: (scores[candidate] / scale - scaled_low)
                        / (scaled_high - scaled_low)
                        for candidate in candidates
                    }
                )

        final_scores = {
            candidate: sum(
                weights[member.member_id] * scores[candidate]
                for member, scores in zip(members, normalized)
            )
            for candidate in candidates
        }
        # One canonical fixed-basket tie rule shared by compiler and validator.
        tie_order = {candidate: index for index, candidate in enumerate(candidates)}
        ranking = sorted(
            candidates,
            key=lambda candidate: (-final_scores[candidate], tie_order[candidate]),
        )
        return {"scores": final_scores, "ranking": ranking}, []

    def _mix_normals(
        self,
        members: list[ForecastMember],
        normals: list[Mapping[str, float]],
        weights: Mapping[str, float],
    ) -> tuple[dict[str, float], list[str]]:
        mean = sum(
            weights[member.member_id] * normal["mean"]
            for member, normal in zip(members, normals)
        )
        variance = math.fsum(
            weights[member.member_id]
            * (normal["sd"] ** 2 + (normal["mean"] - mean) ** 2)
            for member, normal in zip(members, normals)
        )
        variance = max(self.min_sd**2, variance)
        sd = math.sqrt(variance)
        warnings: list[str] = []
        if self.max_sd is not None and sd > self.max_sd:
            sd = self.max_sd
            warnings.append("ensemble sd clipped to configured maximum")
        return {"mean": mean, "sd": sd}, warnings

    def _clip_normal(
        self,
        request: Any,
        normal: dict[str, float],
        warnings: list[str],
        *,
        label: str = "mean",
    ) -> dict[str, float]:
        minimum = field(request, "minimum", None)
        maximum = field(request, "maximum", None)
        mean = normal["mean"]
        if minimum is not None and mean < float(minimum):
            mean = float(minimum)
            warnings.append(f"{label} clipped to minimum")
        if maximum is not None and mean > float(maximum):
            mean = float(maximum)
            warnings.append(f"{label} clipped to maximum")
        return {"mean": mean, "sd": normal["sd"]}
