from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class EnsembleError(Exception):
    """Base exception for ensemble failures."""


class InvalidEnsembleWeights(EnsembleError):
    """Raised when configured member weights are unusable."""


class InvalidEnsembleMember(EnsembleError):
    """Raised when a member prediction violates its task contract."""


@dataclass(frozen=True)
class ForecastMember:
    member_id: str
    member_type: str
    prediction: dict[str, Any]
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalPrediction:
    round_id: str
    target_type: str
    prediction: dict[str, Any]
    members: list[ForecastMember]
    weights: dict[str, float]
    ensemble_version: str
    warnings: list[str]
    degraded: bool = False
