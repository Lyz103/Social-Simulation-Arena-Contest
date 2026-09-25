"""Official metric conventions, chronological selection and held-out reporting.

    Formula reference: Social-Atoms/social-sim-arena e80c8b42c157dadc018441da2a9691b44b20affc.
    Normal quantiles use stdlib NormalDist; compared with the arena's rational
    approximation within 1e-7 (not claimed byte-identical).
"""

from dataclasses import replace
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


def score_records(records):
    """Evaluate held-out predictions; never calls a forecaster with the truth."""
    results = []
    for record in records:
        req = ForecastRequest.from_dict(record["request"])
        loss = score_prediction(req, record["prediction"], record["truth"])
        null = record.get("persistence")
        null_loss = score_prediction(req, null, record["truth"]) if null else None
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
                "loss": loss,
                "persistence_loss": null_loss,
                "skill": (
                    None
                    if null_loss is None or null_loss == 0
                    else 1 - loss / null_loss
                ),
                "marginal_interval_coverage": interval_coverage,
            }
        )
    groups = {}
    for row in results:
        groups.setdefault(row["metric"], []).append(row)
    summary = {}
    for metric, rows in groups.items():
        paired = [r for r in rows if r["persistence_loss"] is not None]
        den = sum(r["persistence_loss"] for r in paired)
        summary[metric] = {
            "count": len(rows),
            "mean_loss": fmean(r["loss"] for r in rows),
            "paired_count": len(paired),
            "aggregate_skill": (
                1 - sum(r["loss"] for r in paired) / den if den else None
            ),
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
    return {"records": results, "by_metric": summary, "record_hash": digest(records)}


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
):
    """Chronological grid selection; test truth is never used to select settings.

    Cases contain precomputed member forecasts. Weights/scales are selected on
    validation only, then frozen and evaluated on the held-out test partition.
    Different metrics must be calibrated in separate calls.
    """
    if parse_instant(validation_start) >= parse_instant(test_start):
        raise ValueError("validation_start must precede test_start")
    cases = sorted(cases, key=lambda c: parse_instant(c["request"]["asof"]))
    if not cases:
        raise ValueError("calibration dataset empty")
    requests = [ForecastRequest.from_dict(c["request"]) for c in cases]
    if len({metric_name(r) for r in requests}) != 1:
        raise ValueError("calibrate each task metric separately")
    groups = {(r.unit, r.source_name, r.population, r.cells) for r in requests}
    if len(groups) != 1:
        raise ValueError(
            "calibrate compatible units, source, population and cell definitions separately"
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
            loss = fmean(
                score_prediction(req, predict(case, req, weights, scale), case["truth"])
                for case, req in validation
            )
            candidates.append(
                {"weights": dict(weights), "sd_scale": scale, "validation_loss": loss}
            )
    best = min(candidates, key=lambda c: (c["validation_loss"], digest(c)))
    test_records = []
    for case, req in test:
        prediction = predict(case, req, best["weights"], best["sd_scale"])
        null = next(
            (
                m["prediction"]
                for m in case["members"]
                if m["member_type"] == "persistence"
            ),
            None,
        )
        test_records.append(
            {
                "request": req.to_dict(),
                "prediction": prediction,
                "truth": case["truth"],
                "persistence": null,
            }
        )
    return {
        "selected": best,
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
