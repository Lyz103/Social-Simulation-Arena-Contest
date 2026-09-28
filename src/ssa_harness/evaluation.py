"""Official metric conventions, chronological selection and held-out reporting.

    Formula reference: Social-Atoms/social-sim-arena e80c8b42c157dadc018441da2a9691b44b20affc.
    Normal quantiles use stdlib NormalDist; compared with the arena's rational
    approximation within 1e-7 (not claimed byte-identical).
"""

from functools import lru_cache
from hashlib import sha256
import math
from statistics import NormalDist, fmean

from .contracts import ForecastRequest
from .ensemble import EnsembleEngine, ForecastMember
from .prompt import PredictionValidator
from .retrieval.utils import digest, parse_instant, source_time_upper


def crps_normal(mean, sd, truth):
    if not all(math.isfinite(x) for x in (mean, sd, truth)) or sd <= 0:
        raise ValueError("finite mean/truth and positive sd required")
    z = (truth - mean) / sd
    return sd * (
        z * math.erf(z / math.sqrt(2))
        + math.sqrt(2 / math.pi) * math.exp(-z * z / 2)
        - 1 / math.sqrt(math.pi)
    )


@lru_cache(maxsize=128)
def _profile_grid(dimension, draws=400, seed=20260811):
    order = sorted(
        range(draws), key=lambda i: sha256(f"{seed}:{dimension}:{i}".encode()).digest()
    )
    return tuple(NormalDist().inv_cdf((i + 0.5) / draws) for i in order)


def energy_normal(cells, truth, order):
    if not order or set(cells) != set(order) or set(truth) != set(order):
        raise ValueError("energy score requires the complete ordered cell vector")
    columns = [
        [cells[c]["mean"] + cells[c]["sd"] * z for z in _profile_grid(i)]
        for i, c in enumerate(order)
    ]
    samples = list(zip(*columns))
    target = [float(truth[c]) for c in order]
    if not all(math.isfinite(x) for x in target):
        raise ValueError("non-finite profile truth")

    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    n = len(samples)
    first = sum(distance(x, target) for x in samples) / n
    pairs = sum(
        distance(samples[i], samples[j]) for i in range(n) for j in range(i + 1, n)
    )
    return first - pairs / (n * n)


def rbo_loss(predicted, truth, p=0.9):
    if (
        not 0 < p < 1
        or len(predicted) != len(truth)
        or not predicted
        or len(set(predicted)) != len(predicted)
        or len(set(truth)) != len(truth)
    ):
        raise ValueError("RBO requires equal-length unique nonempty lists and 0<p<1")
    numerator = denominator = 0
    for i in range(len(predicted)):
        weight = (1 - p) * p**i
        numerator += (
            weight * len(set(predicted[: i + 1]) & set(truth[: i + 1])) / (i + 1)
        )
        denominator += weight
    return 1 - numerator / denominator


def kendall_loss(predicted, truth):
    if (
        len(predicted) < 2
        or len(set(predicted)) != len(predicted)
        or len(set(truth)) != len(truth)
        or set(predicted) != set(truth)
    ):
        raise ValueError("Kendall requires permutations of the same set")
    rank = {x: i for i, x in enumerate(truth)}
    n = len(predicted)
    wrong = sum(
        rank[predicted[i]] > rank[predicted[j]]
        for i in range(n)
        for j in range(i + 1, n)
    )
    return wrong / (n * (n - 1) / 2)


def score_prediction(request, prediction, truth):
    pred = PredictionValidator().parse(prediction, request)
    if request.target_type == "continuous_normal":
        return crps_normal(pred["mean"], pred["sd"], float(truth))
    if request.target_type == "profile_energy":
        return energy_normal(pred["cells"], truth, list(request.cells))
    from .ranking import normalize_title

    actual = [normalize_title(item, request.ranking_spec) for item in truth]
    if request.ranking_spec.get("items"):
        return kendall_loss(pred["ranking"], actual)
    return rbo_loss(pred["ranking"], actual, request.ranking_spec.get("rbo_p", 0.9))


def metric_name(request):
    return {"continuous_normal": "crps", "profile_energy": "energy"}.get(
        request.target_type,
        "kendall" if request.ranking_spec.get("items") else "rbo_loss",
    )


def _comparison_group(request):
    """Raw losses are comparable only within a defined forecasting target."""
    return {
        "source_name": request.source_name,
        "unit": request.unit,
        "population": request.population,
        "metric": request.metric,
        "cells": list(request.cells),
        "scoring_metric": metric_name(request),
    }


def _skill(loss, persistence_loss):
    if persistence_loss is None:
        return None
    # The arena assigns zero skill when the null has zero loss, including
    # imperfect predictions against a perfect ranking null.
    return 0.0 if persistence_loss == 0 else 1 - loss / persistence_loss


def _score_summary(rows):
    paired = [r for r in rows if r["persistence_loss"] is not None]
    den = sum(r["persistence_loss"] for r in paired)
    group_count = len({digest(row["group"]) for row in rows})
    return {
        "count": len(rows),
        "mean_loss": fmean(r["loss"] for r in rows),
        "raw_loss_comparable": group_count == 1,
        "group_count": group_count,
        "paired_count": len(paired),
        "mean_skill": fmean(r["skill"] for r in paired) if paired else None,
        "aggregate_skill": (
            1 - sum(r["loss"] for r in paired) / den if den else None
        ),
        "aggregate_skill_kind": "loss_weighted_legacy",
        "mean_marginal_interval_coverage": (
            {
                str(level): fmean(
                    row["marginal_interval_coverage"][str(level)] for row in rows
                )
                for level in (0.5, 0.8, 0.95)
            }
            if rows[0]["marginal_interval_coverage"] is not None
            else None
        ),
    }


def score_records(records):
    """Evaluate held-out predictions; never calls a forecaster with the truth.

    ``mean_skill`` is the arena's arithmetic mean of per-round skills. The
    legacy ``aggregate_skill`` ratio weights rounds by null loss, so it is not
    the leaderboard objective. Compare raw losses using ``by_group``; the
    retained ``by_metric.mean_loss`` can mix targets, and is explicitly marked
    noncomparable in that case. Missing nulls are excluded, not scored as zero.
    """
    results = []
    for record in records:
        req = ForecastRequest.from_dict(record["request"])
        loss = score_prediction(req, record["prediction"], record["truth"])
        null = record.get("persistence")
        null_loss = (
            score_prediction(req, null, record["truth"]) if null is not None else None
        )
        interval_coverage = None
        if req.target_type != "ranking_list":
            validated = PredictionValidator().parse(record["prediction"], req)
            if req.target_type == "continuous_normal":
                pairs = [(validated, float(record["truth"]))]
            else:
                pairs = [
                    (validated["cells"][c], float(record["truth"][c]))
                    for c in req.cells
                ]
            interval_coverage = {
                str(level): sum(
                    abs(y - normal["mean"])
                    <= NormalDist().inv_cdf((1 + level) / 2) * normal["sd"]
                    for normal, y in pairs
                )
                / len(pairs)
                for level in (0.5, 0.8, 0.95)
            }
        results.append(
            {
                "round_id": req.round_id,
                "metric": metric_name(req),
                "group": _comparison_group(req),
                "loss": loss,
                "persistence_loss": null_loss,
                "persistence_source": (
                    record.get("persistence_source", "record.persistence")
                    if null is not None
                    else None
                ),
                "skill": _skill(loss, null_loss),
                "marginal_interval_coverage": interval_coverage,
            }
        )
    groups = {}
    for row in results:
        groups.setdefault(row["metric"], []).append(row)
    summary = {metric: _score_summary(rows) for metric, rows in groups.items()}
    comparable = {}
    for row in results:
        comparable.setdefault(digest(row["group"]), []).append(row)
    paired = [row for row in results if row["skill"] is not None]
    return {
        "records": results,
        "mean_skill": fmean(row["skill"] for row in paired) if paired else None,
        "paired_count": len(paired),
        "by_metric": summary,
        "by_group": [
            {**rows[0]["group"], **_score_summary(rows)}
            for _, rows in sorted(comparable.items())
        ],
        "record_hash": digest(records),
    }


def walk_forward(cases, predictor):
    """Snapshot/backtester seam: predictor receives only ForecastRequest."""
    ordered = sorted(cases, key=lambda c: parse_instant(c["request"]["asof"]))
    records = []
    for case in ordered:
        req = ForecastRequest.from_dict(case["request"])
        if source_time_upper(case["truth_available_at"]) <= parse_instant(req.asof):
            raise ValueError("target truth was already available at forecast cutoff")
        result = predictor(req)
        prediction = (
            result["final"]["prediction"]
            if isinstance(result, dict) and "final" in result
            else result
        )
        records.append(
            {
                "request": req.to_dict(),
                "prediction": prediction,
                "truth": case["truth"],
                **(
                    {"persistence": case["persistence"]}
                    if "persistence" in case
                    else {}
                ),
            }
        )
    return {"records": records, "evaluation": score_records(records)}


def calibrate(
    cases,
    *,
    validation_start,
    test_start,
    weight_grid=None,
    sd_scales=(0.75, 1.0, 1.5, 2.0),
    objective="mean_loss",
):
    """Chronological grid selection; test truth is never used to select settings.

    Cases contain precomputed member forecasts. Weights/scales are selected on
    validation only, then frozen and evaluated on the held-out test partition.
    Different target metrics must be calibrated in separate calls. The default
    ``mean_loss`` preserves prior behavior. ``mean_skill`` maximizes official
    mean skill and requires explicit official null forecasts in
    ``case.persistence`` for validation and test; ensemble persistence members
    are not silently treated as official nulls. For the legacy loss objective,
    explicit case persistence still takes priority in test reporting.
    """
    if parse_instant(validation_start) >= parse_instant(test_start):
        raise ValueError("validation_start must precede test_start")
    if objective not in ("mean_loss", "mean_skill"):
        raise ValueError("objective must be mean_loss or mean_skill")
    cases = sorted(cases, key=lambda c: parse_instant(c["request"]["asof"]))
    if not cases:
        raise ValueError("calibration dataset empty")
    requests = [ForecastRequest.from_dict(c["request"]) for c in cases]
    if len({metric_name(r) for r in requests}) != 1:
        raise ValueError("calibrate each task metric separately")
    groups = {digest(_comparison_group(r)) for r in requests}
    if len(groups) != 1:
        raise ValueError(
            "calibrate compatible units, source, population, target metric and cell definitions separately"
        )
    validation, test = [], []
    for case, req in zip(cases, requests):
        available = source_time_upper(case["truth_available_at"])
        if available <= parse_instant(req.asof):
            raise ValueError("truth must be unavailable at forecast time")
        date = parse_instant(req.asof)
        if parse_instant(validation_start) <= date < parse_instant(test_start):
            if available >= parse_instant(test_start):
                raise ValueError("validation outcome not known before test start")
            validation.append((case, req))
        elif date >= parse_instant(test_start):
            test.append((case, req))
    if not validation or not test:
        raise ValueError("nonempty chronological validation and test sets required")
    if objective == "mean_skill" and any(
        case.get("persistence") is None for case, _ in validation + test
    ):
        raise ValueError(
            "mean_skill requires explicit official case.persistence for validation and test"
        )
    # Only validation truth/nulls enter candidate selection. Test truth and
    # nulls are consumed after settings have been frozen.
    validation_null_losses = [
        score_prediction(req, case["persistence"], case["truth"])
        if case.get("persistence") is not None
        else None
        for case, req in validation
    ]
    grids = weight_grid or [
        {"llm": 1.0, "persistence": 0.0, "recent_mean": 0.0},
        {"llm": 0.6, "persistence": 0.25, "recent_mean": 0.15},
        {"llm": 0.3, "persistence": 0.5, "recent_mean": 0.2},
        {"llm": 0.0, "persistence": 1.0, "recent_mean": 0.0},
    ]
    if requests[0].target_type == "ranking_list":
        sd_scales = (1.0,)
    if not sd_scales or any(not math.isfinite(s) or s <= 0 for s in sd_scales):
        raise ValueError("invalid calibration scales")

    def predict(case, req, weights, scale):
        members = [ForecastMember(**m) for m in case["members"]]
        result = (
            EnsembleEngine(weights=weights)
            .combine(req, [], baselines=members)
            .prediction
        )
        if req.target_type != "ranking_list":
            normals = (
                result["cells"].values()
                if req.target_type == "profile_energy"
                else [result]
            )
            for normal in normals:
                normal["sd"] *= scale
        return PredictionValidator().parse(result, req)

    candidates = []
    for weights in grids:
        for scale in sd_scales:
            losses = [
                score_prediction(req, predict(case, req, weights, scale), case["truth"])
                for case, req in validation
            ]
            mean_skill = (
                fmean(
                    _skill(loss, null_loss)
                    for loss, null_loss in zip(losses, validation_null_losses)
                )
                if all(loss is not None for loss in validation_null_losses)
                else None
            )
            candidates.append(
                {
                    "weights": dict(weights),
                    "sd_scale": scale,
                    "validation_loss": fmean(losses),
                    "validation_mean_skill": mean_skill,
                }
            )
    best = min(
        candidates,
        key=lambda c: (
            c["validation_loss"]
            if objective == "mean_loss"
            else -c["validation_mean_skill"],
            digest(c),
        ),
    )
    test_records = []
    for case, req in test:
        prediction = predict(case, req, best["weights"], best["sd_scale"])
        if "persistence" in case:
            null = case["persistence"]
            null_source = "case.persistence"
        else:
            null = next(
                (
                    m["prediction"]
                    for m in case["members"]
                    if m["member_type"] == "persistence"
                ),
                None,
            )
            null_source = "member.persistence (not an official null)"
        test_records.append(
            {
                "request": req.to_dict(),
                "prediction": prediction,
                "truth": case["truth"],
                "persistence": null,
                "persistence_source": null_source if null is not None else None,
            }
        )
    return {
        "selected": best,
        "objective": objective,
        "validation_candidates": candidates,
        "validation_count": len(validation),
        "test_count": len(test),
        "validation_start": validation_start,
        "test_start": test_start,
        "dataset_hash": digest(cases),
        "test": score_records(test_records),
        "test_records": test_records,
        "note": "Offline selection result; predictive performance is only established on supplied held-out cases.",
    }
