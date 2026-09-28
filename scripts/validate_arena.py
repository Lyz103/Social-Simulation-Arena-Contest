"""Prepare frozen public Arena cases, predict without labels, then score separately.

Run from the repository root. Source captures live in data/arena-validation.
No API credentials are accepted in files or command-line arguments.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

from ssa_harness.cli import load_pipeline
from ssa_harness.contracts import ForecastRequest
from ssa_harness.ensemble import BaselineFactory, EnsembleEngine
from ssa_harness.evaluation import score_records
from ssa_harness.retrieval.utils import digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/arena-validation"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_prediction(path, value):
    """A new experiment needs a new directory; do not replace prior forecasts."""
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f"prediction differs from existing output: {path}; use a new --data-dir and archive identity")
    write(path, value)


def baseline_predictions(req, suite):
    factory = BaselineFactory() if suite == "legacy" else BaselineFactory(
        numeric_members=("persistence", "recent_mean", "ewma", "local_trend"),
        sd_method="rolling_residual",
    )
    members = factory.build(req)
    config_hash = digest({"baseline": factory.identity(), "ensemble_version": EnsembleEngine.VERSION})
    prefix = "" if suite == "legacy" else "rolling_"
    predictions = {prefix + m.member_id: m.prediction for m in members}
    if suite == "legacy":
        predictions["baseline_ensemble"] = EnsembleEngine().combine(req, [], members).prediction
    return predictions, config_hash


def verify_legacy_exports():
    """Upgrade old export envelopes only after reproducing their input/forecast binding.

    Model values are checked against archived requests and predictions. Baselines
    are recomputed on the named inputs and must agree exactly. Never calls LLMs.
    """
    inputs = {r["round_id"]: ForecastRequest.from_dict(r) for r in read(DATA / "inputs.json")}
    count = 0
    for directory in sorted((DATA / "predictions").iterdir()):
        for f in sorted(directory.glob("*.json")):
            exported = read(f)
            if exported.get("input_hash") or exported.get("config_hash"):
                continue
            req = inputs[exported["round_id"]]
            if directory.name == "deepseek":
                run = read(DATA / "runs" / (req.round_id + ".json"))
                if run["input_hash"] != req.request_hash or ForecastRequest.from_dict(run["request"]).request_hash != req.request_hash:
                    raise ValueError(f"legacy run input mismatch: {f}")
                expected = {"ensemble": run["final"]["prediction"]}
                for member in run["final"]["members"]:
                    if member["member_type"] == "llm":
                        expected["llm"] = member["prediction"]
                config_hash = run["pipeline_config_hash"]
            elif directory.name in ("baseline", "expanded-baseline"):
                expected, config_hash = baseline_predictions(req, "legacy" if directory.name == "baseline" else "expanded")
            else:
                raise ValueError(f"unrecognized legacy prediction directory: {directory}")
            if exported["predictions"] != expected:
                raise ValueError(f"legacy forecast failed verification: {f}")
            write(f, {**exported, "input_hash": req.request_hash, "config_hash": config_hash})
            count += 1
    print(f"Verified and added identity to {count} legacy exports; forecast values unchanged.")


def instant(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def prepare(source_dir=None):
    source_dir = Path(source_dir) if source_dir else DATA
    site = read(source_dir / "site-data.json")
    tree = read(source_dir / "github-tree.json")
    inputs, labels, excluded, provenance = [], {}, [], []
    for r in site["rounds"]:
        if r["status"] != "resolved":
            continue
        rid = r["round_id"]
        file = source_dir / "upstream/locks" / (rid + ".json")
        reason = None
        snap = read(file) if file.exists() else {}
        cutoff = instant(r["lock_at"]) - timedelta(hours=24)
        if not r.get("scoreable"):
            reason = "officially unscoreable"
        elif snap.get("corrected"):
            reason = "historically corrected snapshot excluded"
        elif not snap.get("answer_frozen_at"):
            reason = "no explicit answer-window frozen snapshot"
        elif instant(snap["answer_frozen_at"]) > cutoff:
            reason = "snapshot later than call-window opening"
        elif r["target_type"] == "profile_energy" and not snap.get("answer_history_by_cell"):
            reason = "no complete frozen profile history"
        elif r["target_type"] == "ranking_list":
            reason = "ranking adapter not included in this validation batch"
        if reason:
            excluded.append({"round_id": rid, "reason": reason})
            continue
        frozen = snap["answer_frozen_at"]

        def point(p, value):
            assert instant(p["date"] + "T00:00:00Z" if len(p["date"]) == 10 else p["date"]) <= cutoff
            return {"date": p["date"], "value": value, "available_at": frozen,
                    "provenance": "official_answer_window_snapshot_asserted"}

        if r["target_type"] == "continuous_normal":
            history = [point(p, p["value"]) for p in snap["answer_history"][-24:]]
            null = {k: r["baselines"]["persistence"][k] for k in ("mean", "sd")}
            truth = r["resolution"]["value"]
        else:
            history = [point(p, {cell: p["value"]}) for cell in r["cells"]
                       for p in snap["answer_history_by_cell"][cell][-12:]]
            null = {"cells": {c: {k: r["profile"]["baselines"]["persistence"][c][k]
                                       for k in ("mean", "sd")} for c in r["cells"]}}
            truth = r["resolution"]["outcome"]
        assert instant(r["release_at"]) > cutoff
        request = ForecastRequest(
            round_id=rid, question=r["question"], target_type=r["target_type"],
            unit=r["unit"], asof=cutoff.isoformat(), deadline=r["lock_at"],
            release_at=r["release_at"], history=history, cells=r.get("cells", []),
            source_name=r["tracker"], metric=r["series"], domain=r.get("domain", "general"),
            metadata={"snapshot_commit": tree["sha"], "snapshot_frozen_at": frozen},
        )
        inputs.append(request.to_dict())
        labels[rid] = {"truth": truth, "persistence": null, "resolution": r["resolution"]}
        provenance.append({"round_id": rid, "url": "https://raw.githubusercontent.com/Social-Atoms/social-sim-arena/" + tree["sha"] + "/locks/" + rid + ".json",
                           "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "frozen_at": frozen})
    inputs.sort(key=lambda r: (r["asof"], r["round_id"]))
    write(DATA / "inputs.json", inputs)
    write(DATA / "labels.json", labels)
    write(DATA / "manifest.json", {
        "site_generated_at": site["generated_at"], "prepared_at": datetime.now(timezone.utc).isoformat(),
        "commit": tree["sha"], "status_counts": dict(Counter(r["status"] for r in site["rounds"])),
        "selected": len(inputs), "types": dict(Counter(r["target_type"] for r in inputs)),
        "selection": "All resolved scoreable rounds with explicit answer-window snapshots at or before lock minus 24h; no score-based selection.",
        "limitations": ["Snapshots downloaded now; historical availability is asserted by official archive, not independently verified.",
                        "Historical LLM knowledge contamination cannot be excluded.",
                        "No live search; no tuning on these labels; ranking rounds excluded."],
        "snapshots": provenance, "excluded": excluded,
    })
    print("Prepared", len(inputs), "cases; excluded", len(excluded), flush=True)


def predict(config=None, limit=None, baseline_suite="legacy"):
    # Deliberately never open site-data.json or labels.json in the prediction stage.
    requests = read(DATA / "inputs.json")
    if limit:
        requests = requests[:limit]
    pipeline, _ = load_pipeline(config) if config else (None, None)
    name = "deepseek" if pipeline else "baseline" if baseline_suite == "legacy" else "expanded-baseline"
    for i, raw in enumerate(requests):
        req = ForecastRequest.from_dict(raw)
        destination = DATA / "predictions" / name / (req.round_id + ".json")
        if pipeline:
            result = pipeline.run(req)
            config_hash = result["pipeline_config_hash"]
            predictions = {"ensemble": result["final"]["prediction"]}
            for member in result["final"]["members"]:
                if member["member_type"] == "llm":
                    predictions["llm"] = member["prediction"]
            write_prediction(DATA / "runs" / (req.round_id + ".json"), result)
        else:
            predictions, config_hash = baseline_predictions(req, baseline_suite)
        write_prediction(destination, {"round_id": req.round_id, "input_hash": req.request_hash,
                                       "config_hash": config_hash, "predictions": predictions})
        print(f"{i+1}/{len(requests)} {name} {req.round_id}", flush=True)


def score():
    labels = read(DATA / "labels.json")
    strategies = {}
    configuration_by_directory = {}
    for raw in read(DATA / "inputs.json"):
        rid = raw["round_id"]
        req = ForecastRequest.from_dict(raw)
        truth, null = labels[rid]["truth"], labels[rid]["persistence"]
        predictions = {"official_persistence": null}
        for directory in sorted((DATA / "predictions").iterdir()):
            f = directory / (rid + ".json")
            if f.exists():
                exported = read(f)
                if exported.get("round_id") != rid or exported.get("input_hash") != req.request_hash:
                    raise ValueError(f"prediction/input identity mismatch: {f}")
                config_hash = exported.get("config_hash")
                if not config_hash or configuration_by_directory.setdefault(directory.name, config_hash) != config_hash:
                    raise ValueError(f"mixed or missing prediction config identity: {f}")
                if directory.name == "deepseek":
                    run = read(DATA / "runs" / (rid + ".json"))
                    if run["input_hash"] != req.request_hash or run["pipeline_config_hash"] != config_hash:
                        raise ValueError(f"prediction/archive identity mismatch: {f}")
                overlap = predictions.keys() & exported["predictions"].keys()
                if overlap:
                    raise ValueError(f"duplicate prediction strategies: {sorted(overlap)}")
                predictions.update(exported["predictions"])
        for name, pred in predictions.items():
            strategies.setdefault(name, []).append({"request": raw, "truth": truth,
                                                    "persistence": null, "prediction": pred})
    results = {name: score_records(rows) for name, rows in strategies.items()}
    # Report each metric AND source/target series; never average heterogeneous raw units.
    grouped = {}
    for name, rows in strategies.items():
        for row in rows:
            key = row["request"]["metric"]
            grouped.setdefault(name, {}).setdefault(key, []).append(row)
    per_series = {name: {key: score_records(rows)["by_metric"] for key, rows in groups.items()}
                  for name, groups in grouped.items()}
    write(DATA / "evaluation.json", {"strategies": results, "by_series": per_series,
                                    "warning": "Do not interpret cross-series mean CRPS as an official leaderboard score; units and target definitions differ."})
    counts = Counter(r["target_type"] for r in read(DATA / "inputs.json"))
    manifest = read(DATA / "manifest.json") if (DATA / "manifest.json").exists() else {}
    data_arg = str(DATA.relative_to(ROOT) if DATA.is_relative_to(ROOT) else DATA)
    report = ["# Arena 真实题目验证", "", f'已结算题型与数量：{dict(counts)}。',
              "只使用官方 answer-window 历史，准备的输入最多数值 24 点、每群体 12 点；模型实际可见窗口由运行归档的 context.config 决定。预测阶段不读取答案和其他参赛者预测。",
              "本批为已分析过的开发集，不能作为独立测试集。统计候选使用预先配置的参数，并不依据本批结果自动选优。",
              "下表 mean skill 按官方逐题平均口径计算，0 为 Persistence；aggregate_skill 仅保留为历史兼容字段，不能当官方均分。", "",
              "| 策略 | 题型 | 题数 | mean skill ×100 | 胜 / 平 / 负（对官方 Persistence） |",
              "|---|---|---:|---:|---|"]
    for name, result in results.items():
        for metric in ("crps", "energy"):
            rows = [r for r in result["records"] if r["metric"] == metric]
            if not rows:
                continue
            wins = sum(r["loss"] < r["persistence_loss"] - 1e-9 for r in rows)
            ties = sum(abs(r["loss"] - r["persistence_loss"]) <= 1e-9 for r in rows)
            skill = result["by_metric"][metric]["mean_skill"]
            report.append(f"| {name} | {metric} | {len(rows)} | {100 * skill:.2f} | {wins} / {ties} / {len(rows)-wins-ties} |")
    report += ["", "## 数值题明细", "", "mean ± sd 不是置信区间；CRPS 越低越好。", "",
               "| 题目 | 真值 | 基线集成 mean ± sd | 官方 Persistence CRPS | 基线集成 CRPS | LLM mean ± sd | LLM CRPS | 最终集成 CRPS |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    indexed = {name: {r["request"]["round_id"]: r for r in rows} for name, rows in strategies.items()}
    scored = {name: {r["round_id"]: r for r in result["records"]} for name, result in results.items()}
    for raw in read(DATA / "inputs.json"):
        if raw["target_type"] != "continuous_normal":
            continue
        rid = raw["round_id"]
        b = indexed.get("baseline_ensemble", {}).get(rid)
        if not b:
            continue
        p = b["prediction"]
        llm = indexed.get("llm", {}).get(rid)
        llm_text = f'{llm["prediction"]["mean"]:.3f} ± {llm["prediction"]["sd"]:.3f}' if llm else "待运行"
        def loss(name):
            row = scored.get(name, {}).get(rid)
            return f'{row["loss"]:.4f}' if row else "—"
        report.append(f'| {rid} | {b["truth"]} | {p["mean"]:.3f} ± {p["sd"]:.3f} | {loss("official_persistence")} | {loss("baseline_ensemble")} | {llm_text} | {loss("llm")} | {loss("ensemble")} |')
    report += ["", "## 分群题明细", "", "Energy score 越低越好，每题有 16 个 cell。", "",
               "| 题目 | 官方 Persistence | 基线集成 | 纯 LLM | 最终集成 |", "|---|---:|---:|---:|---:|"]
    for raw in read(DATA / "inputs.json"):
        if raw["target_type"] == "profile_energy":
            rid = raw["round_id"]
            values = []
            for name in ("official_persistence", "baseline_ensemble", "llm", "ensemble"):
                row = scored.get(name, {}).get(rid)
                values.append(f'{row["loss"]:.4f}' if row else "待运行")
            report.append("| " + rid + " | " + " | ".join(values) + " |")
    runs = [read(p) for p in sorted((DATA / "runs").glob("*.json"))]
    if runs:
        usage = Counter()
        models = set()
        for run in runs:
            for reply in run["model_runs"]:
                models.add(reply["model"])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[key] += reply.get("usage", {}).get(key, 0)
        summary = {"successful_rounds": len(runs), "returned_models": sorted(models),
                   "successful_run_usage": dict(usage),
                   "note": "Usage covers saved successful runs, including returned length-retry attempts when archived by the client; excludes failed runs and superseded configurations. Not a billing total.",
                   "pipeline_config_hashes": sorted({r["pipeline_config_hash"] for r in runs}),
                   "context_configs": [json.loads(s) for s in sorted({json.dumps(r["context"]["config"], sort_keys=True) for r in runs})],
                   "degraded_rounds": [r["request"]["round_id"] for r in runs if r["final"]["degraded"]]}
        write(DATA / "model-summary.json", summary)
        report += ["", "## 模型运行", "", f'返回模型：{", ".join(sorted(models))}；成功题数：{len(runs)}。',
                   f'成功运行记录中的 token：{usage["total_tokens"]}（包含客户端已归档的长度重试；不包含失败轮次及被替代配置的请求，不是账单总额）。',
                   "最终成员权重和是否降级以逐题 runs/*.json 的 final 字段为准。"]
    report += ["", "## 验证边界", "",
               "- 本地回溯验证，不是比赛提交或独立测试集成绩；不排除模型训练数据包含历史答案。",
               "- 原始快照目前从官方仓库下载，历史可用时间依赖官方归档声明。",
               f'- {len(manifest.get("excluded", []))} 道已结算题因冻结历史不满足本批条件而排除，原因见 manifest.json。',
               "- 本批没有排名题，也没有实时检索；结果不能代表完整线上研究流水线。",
               "- 官方 Persistence 的均值与输入最后观测全部一致；仓库 baseline 的标准差算法不同。",
               "", "## 复现", "", "```bash",
               f".venv/bin/python scripts/validate_arena.py predict --data-dir {data_arg}",
               "# 可选的 EWMA、局部趋势及滚动残差统计候选：",
               f".venv/bin/python scripts/validate_arena.py predict --baseline-suite expanded --data-dir {data_arg}",
               "# 将 SSA_LLM_API_KEY 设置在环境中后：",
               f".venv/bin/python scripts/validate_arena.py predict --config <matching-experiment-config.json> --data-dir {data_arg}",
               f".venv/bin/python scripts/validate_arena.py score --data-dir {data_arg}", "```", ""]
    (DATA / "REPORT.md").write_text("\n".join(report))
    for name, result in results.items():
        print(name, json.dumps(result["by_metric"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "predict", "score", "verify-legacy-exports"])
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--baseline-suite", choices=["legacy", "expanded"], default="legacy")
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    if args.stage == "prepare":
        prepare(args.source_dir)
    elif args.stage == "predict":
        predict(args.config, args.limit, args.baseline_suite)
    elif args.stage == "verify-legacy-exports":
        verify_legacy_exports()
    else:
        score()
