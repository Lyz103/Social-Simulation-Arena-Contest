"""Internal task contract and adapters for the pinned SSA Agent API v2."""

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from importlib.resources import files
import json
import math
from typing import Any

from jsonschema import Draft7Validator, FormatChecker

from .retrieval.utils import canonical_json, digest, iso, parse_instant


class ContractError(ValueError):
    pass


def validate_schema(payload, name):
    schema = json.loads(
        files("ssa_harness").joinpath("schemas", name).read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft7Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda e: str(e.path),
    )
    if errors:
        error = errors[0]
        # Do not echo entire payloads or remote bodies in errors.
        raise ContractError(
            f"{name}: invalid {'.'.join(map(str, error.path)) or 'root'} ({error.validator})"
        )


def _strings(value, label, *, allow_empty=True):
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(x, str) or not x.strip() for x in value
    ):
        raise ContractError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value) or (not value and not allow_empty):
        raise ContractError(f"{label} must be unique and nonempty when required")
    return tuple(value)


@dataclass(frozen=True)
class ForecastRequest:
    round_id: str
    question: str
    target_type: str
    unit: str
    asof: str
    deadline: str
    request_id: str = ""
    board_id: str = ""
    forecast_for: str | None = None
    target_start: str | None = None
    target_end: str | None = None
    release_at: str | None = None
    history: tuple[dict, ...] = ()
    cells: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    ranking_spec: dict = field(default_factory=dict)
    minimum: float | None = None
    maximum: float | None = None
    domain: str = "general"
    source_name: str = ""
    population: str = ""
    metric: str = ""
    source_domains: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.round_id, self.question, self.unit)
        ):
            raise ContractError("round_id, question and unit are required")
        boards = {
            "continuous_normal": "topline",
            "profile_energy": "profile",
            "ranking_list": "ranking",
        }
        if not isinstance(self.ranking_spec, dict) or not isinstance(
            self.metadata, dict
        ):
            raise ContractError("ranking_spec and metadata must be objects")
        if self.target_type not in boards:
            raise ContractError("unsupported target_type")
        if self.board_id and self.board_id != boards[self.target_type]:
            raise ContractError("board_id/target_type mismatch")
        object.__setattr__(self, "board_id", boards[self.target_type])
        cutoff, deadline = parse_instant(self.asof), parse_instant(self.deadline)
        if cutoff >= deadline:
            raise ContractError("asof must precede deadline")
        object.__setattr__(self, "asof", iso(cutoff))
        object.__setattr__(self, "deadline", iso(deadline))
        for name in ("cells", "candidates", "source_domains"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if self.target_type == "profile_energy" and len(self.cells) < 2:
            raise ContractError("profile requires at least two named cells")
        if self.target_type == "ranking_list":
            n = self.ranking_spec.get("length")
            if type(n) is not int or not 2 <= n <= 50:
                raise ContractError("ranking length must be 2..50")
            if self.ranking_spec.get("items"):
                items = _strings(
                    self.ranking_spec["items"], "ranking.items", allow_empty=False
                )
                if len(items) != n or (self.candidates and self.candidates != items):
                    raise ContractError(
                        "fixed ranking items must match length and candidates"
                    )
                object.__setattr__(self, "candidates", items)
            p = self.ranking_spec.get("rbo_p", 0.9)
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p < 1:
                raise ContractError("rbo_p must be between zero and one")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
            ):
                raise ContractError("bounds must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ContractError("minimum exceeds maximum")
        for name in ("target_start", "target_end", "release_at"):
            value = getattr(self, name)
            if value is not None:
                from .retrieval.utils import source_time_upper

                source_time_upper(value)
        if self.target_start and self.target_end:
            from .retrieval.utils import source_time_upper

            if source_time_upper(self.target_start) > source_time_upper(
                self.target_end
            ):
                raise ContractError("target_start exceeds target_end")
        if not isinstance(self.history, (tuple, list)) or any(
            not isinstance(x, dict) for x in self.history
        ):
            raise ContractError("history must be timestamped observation objects")
        object.__setattr__(self, "history", tuple(self.history))
        canonical_json(asdict(self))

    @property
    def is_open_ranking(self):
        return self.target_type == "ranking_list" and not bool(
            self.ranking_spec.get("items")
        )

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))

    @property
    def request_hash(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        try:
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise ContractError("invalid ForecastRequest fields") from exc

    @classmethod
    def from_envelope(cls, envelope, *, asof=None, metadata=None):
        """Use the official frozen-context boundary, never observation date as publication time.

        The default cutoff is deadline-24h for the pinned season-0 endpoint
        contract. Explicit asof is supported for a caller with a stricter policy.
        Official context carries an availability assertion, not a trusted timestamp.
        """
        validate_schema(envelope, "agent-api-request.schema.json")
        r, meta = envelope["round"], dict(metadata or {})
        frozen_at = parse_instant(r["lock_at"]) - timedelta(hours=24)
        cutoff = parse_instant(asof) if asof else frozen_at
        ctx = r["context"]
        rows = []

        def point(p, value):
            if not isinstance(p, dict):
                raise ContractError("official history points must be objects")
            row = dict(p)
            row["value"] = value
            # Earlier custom cutoffs exclude context lacking finer availability.
            row.setdefault("available_at", iso(frozen_at))
            row.setdefault("provenance", "official_frozen_context_asserted")
            return row

        kind = r["target_type"]
        if kind == "continuous_normal":
            rows = [point(p, p["value"]) for p in ctx.get("history", [])]
            if not rows and ctx.get("persistence") is not None:
                rows = [point({"date": iso(frozen_at)}, ctx["persistence"])]
        elif kind == "profile_energy":
            for cell, points in ctx.get("history_by_cell", {}).items():
                if cell in r.get("cells", []):
                    rows.extend(point(p, {cell: p["value"]}) for p in points)
        elif kind == "ranking_list":
            rows = [
                point(p, p.get("items", p.get("ranking")))
                for p in ctx.get("recent_weeks", [])
            ]
        meta["history_provenance"] = "official_frozen_context_asserted"
        meta["official_context_asof"] = iso(frozen_at)
        meta["cutoff_policy"] = (
            "endpoint-window-open-v1" if asof is None else "explicit"
        )
        spec = dict(r.get("ranking", {}))
        return cls(
            round_id=r["round_id"],
            request_id=envelope["request_id"],
            board_id=r["board_id"],
            question=r["question"],
            target_type=kind,
            unit=r["unit"],
            asof=iso(cutoff),
            deadline=r["lock_at"],
            history=tuple(rows),
            cells=tuple(r.get("cells", [])),
            candidates=tuple(spec.get("items", [])),
            ranking_spec=spec,
            forecast_for=meta.get("forecast_for"),
            target_start=meta.get("target_start", spec.get("week_start")),
            target_end=meta.get("target_end", spec.get("week_end")),
            release_at=meta.get("release_at"),
            minimum=meta.get("minimum"),
            maximum=meta.get("maximum"),
            domain=meta.get("domain", "general"),
            source_name=meta.get("source_name", ""),
            population=meta.get("population", ""),
            metric=meta.get("metric", ""),
            source_domains=tuple(meta.get("source_domains", [])),
            metadata=meta,
        )

    def to_research_request(self, *, mode="snapshot"):
        from .retrieval.models import ResearchRequest

        return ResearchRequest(
            round_id=self.round_id,
            question=self.question,
            target_type=self.target_type,
            unit=self.unit,
            asof=self.asof,
            deadline=self.deadline,
            mode=mode,
            domain=self.domain,
            source_name=self.source_name,
            population=self.population,
            metric=self.metric,
            source_domains=self.source_domains,
            cells=self.cells,
            ranking_spec=self.ranking_spec,
            history=list(self.history),
            context={"request_hash": self.request_hash},
        )


def response_envelope(request, prediction):
    from .prompt.validators import PredictionValidator

    pred = PredictionValidator().parse(prediction, request)
    if request.target_type == "profile_energy":
        forecast = {"profile": pred["cells"]}
    elif request.target_type == "ranking_list":
        forecast = {"ranking": pred["ranking"]}
    else:
        forecast = pred
    result = {"schema_version": "ssa-agent-api-v2", "forecast": forecast}
    validate_schema(result, "agent-api-response.schema.json")
    return result


def signed_forecast_document(request, prediction, entrant):
    """Build an unsigned file for the arena's official signing tool. Does not submit."""
    fc = response_envelope(request, prediction)["forecast"]
    if request.target_type == "continuous_normal":
        fc = {"topline": fc}
    return {"round_id": request.round_id, "entrant": entrant, **fc}
