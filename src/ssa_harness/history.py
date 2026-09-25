"""Normalize history once; the context and every baseline share this result."""

from dataclasses import dataclass, replace
from collections.abc import Mapping
import math

from .retrieval.utils import digest, iso, parse_instant, source_time_upper


class HistoryError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedHistory:
    rows: tuple[dict, ...]
    warnings: tuple[str, ...]
    history_hash: str


def numeric(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise HistoryError("observation must be a finite number")
    return float(value)


def normalize_history(request):
    cutoff = parse_instant(request.asof)
    selected, warnings = {}, []
    for index, raw in enumerate(request.history):
        row = dict(raw)
        available = row.get("available_at") or row.get("published_at")
        if not available:
            raise HistoryError(f"history[{index}] lacks available_at/published_at")
        available_dt = source_time_upper(available)
        if available_dt > cutoff:
            warnings.append(f"excluded history[{index}]: available after cutoff")
            continue
        # All supplied temporal evidence must agree with the cutoff.
        if any(
            row.get(k) and source_time_upper(row[k]) > cutoff
            for k in ("published_at", "captured_at")
        ):
            warnings.append(f"excluded history[{index}]: source timestamp after cutoff")
            continue
        observed = row.get(
            "observed_at", row.get("date", row.get("week_end", available))
        )
        observed_dt = source_time_upper(observed)
        if observed_dt > cutoff:
            warnings.append(f"excluded history[{index}]: observation after cutoff")
            continue
        value = row.get(
            "value",
            row.get("profile", row.get("cells", row.get("ranking", row.get("items")))),
        )
        if request.target_type == "continuous_normal":
            entries = [("", numeric(value))]
        elif request.target_type == "profile_energy":
            if not isinstance(value, Mapping):
                raise HistoryError("profile history value must be a cell map")
            unknown = set(value) - set(request.cells)
            if unknown:
                raise HistoryError("profile history contains unknown cells")
            entries = [(c, numeric(v)) for c, v in value.items()]
        else:
            from .ranking import normalize_title

            if not isinstance(value, (tuple, list)):
                raise HistoryError("ranking history value must be a list")
            normalized = [normalize_title(v, request.ranking_spec) for v in value]
            if len(set(normalized)) != len(normalized):
                raise HistoryError("ranking history contains duplicate canonical items")
            if request.ranking_spec.get("items") and set(normalized) != set(
                request.candidates
            ):
                raise HistoryError("fixed ranking history candidate mismatch")
            entries = [("", normalized)]
        for cell, val in entries:
            if request.target_type != "ranking_list":
                if (
                    request.minimum is not None
                    and val < request.minimum
                    or request.maximum is not None
                    and val > request.maximum
                ):
                    raise HistoryError("history observation outside declared bounds")
            key = (observed_dt, cell)
            clean = {
                "date": str(observed),
                "observed_at": iso(observed_dt),
                "available_at": iso(available_dt),
                "value": {cell: val} if cell else val,
                "provenance": row.get("provenance", "caller_asserted"),
            }
            for name in ("source", "source_url", "content_hash", "views", "index"):
                if name in row:
                    clean[name] = row[name]
            previous = selected.get(key)
            if (
                previous
                and previous[0] == available_dt
                and previous[1]["value"] != clean["value"]
            ):
                raise HistoryError(
                    "conflicting observations have identical observation and availability time"
                )
            if previous is None or previous[0] < available_dt:
                selected[key] = (available_dt, clean)
    rows = tuple(selected[key][1] for key in sorted(selected))
    return NormalizedHistory(rows, tuple(warnings), digest(rows))


def normalized_request(request):
    history = normalize_history(request)
    meta = dict(request.metadata)
    meta["history_hash"] = history.history_hash
    meta["history_warnings"] = list(history.warnings)
    return replace(request, history=history.rows, metadata=meta), history
