"""Unified CLI. All network/baseline-only policies are explicit in configuration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .archive import RunArchive
from .context import ContextBuilder, ContextConfig
from .contracts import ForecastRequest
from .ensemble import EnsembleEngine, BaselineFactory
from .llm import ChatCompletionsClient, ReplayClient
from .pipeline import ForecastPipeline, PipelineConfig
from .prompt import ForecastPromptCompiler
from .retrieval import (
    ArchiveStore,
    ResearchEngine,
    SnapshotRetriever,
    TavilyRetriever,
    EngineConfig,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    dest = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if read(dest) == value:
            return
        raise ValueError("refusing to overwrite different output: " + str(dest))
    with dest.open("x", encoding="utf-8") as f:
        f.write(text)


def load_pipeline(path):
    file = Path(path).resolve()
    config = read(file)

    def resolve(value):
        return str((file.parent / value).resolve())

    settings = PipelineConfig(**config.get("pipeline", {}))
    clients = {}
    for row in config.get("models", []):
        row = dict(row)
        name = row.pop("member_id")
        if name in clients:
            raise ValueError("duplicate model member_id")
        kind = row.pop("kind", "chat_completions")
        if kind == "replay":
            clients[name] = ReplayClient(read(resolve(row["responses"])), name=name)
        elif kind == "chat_completions":
            clients[name] = ChatCompletionsClient(**row)
        else:
            raise ValueError("unknown model kind")
    research = None
    if settings.research_policy == "required":
        source = config["research"]
        if source["provider"] == "snapshot":
            retriever = SnapshotRetriever(resolve(source["corpus"]))
        elif source["provider"] == "tavily":
            retriever = TavilyRetriever()
        elif source["provider"] == "structured":
            from .sources import StructuredRetriever

            retriever = StructuredRetriever([resolve(p) for p in source["packages"]])
        elif source["provider"] == "civiqs":
            from .source_adapters import CiviqsRetriever

            retriever = CiviqsRetriever(
                tracker=source.get("tracker", "approve_president_trump_2025"),
                filters=source.get("filters"),
                api_base=source.get("api_base", "https://civiqs.com/results"),
                user_agent=source.get(
                    "user_agent", "MIT-SSA-research/0.1 (contact project maintainer)"
                ),
                timeout=source.get("timeout", 60),
            )
        elif source["provider"] == "yougov":
            from .source_adapters import YouGovRetriever

            retriever = YouGovRetriever(
                tracker=source.get("tracker", "donald-trump-approval"),
                api_base=source.get(
                    "api_base", "https://api-test.yougov.com/public-data/v5/us/trackers"
                ),
                user_agent=source.get(
                    "user_agent", "MIT-SSA-research/0.1 (contact project maintainer)"
                ),
                timeout=source.get("timeout", 60),
                scored_cells=source.get("scored_cells"),
            )
        else:
            raise ValueError("unknown research provider")
        research = ResearchEngine(
            retriever=retriever,
            store=ArchiveStore(
                resolve(source.get("db", "../data/research/bundles.sqlite3"))
            ),
            config=EngineConfig(**source.get("config", {})),
        )
    ensemble_args = dict(config.get("ensemble", {}))
    baseline_args = ensemble_args.pop("baseline", {})
    ensemble_args.setdefault("failure_policy", "drop_invalid")
    pipeline = ForecastPipeline(
        archive=RunArchive(
            resolve(config.get("run_db", "../data/runs/forecast.sqlite3"))
        ),
        clients=clients,
        research_engine=research,
        context_builder=ContextBuilder(ContextConfig(**config.get("context", {}))),
        compiler=ForecastPromptCompiler(**config.get("prompt", {})),
        ensemble=EnsembleEngine(
            baseline_factory=BaselineFactory(**baseline_args), **ensemble_args
        ),
        config=settings,
        source_packages=[read(resolve(p)) for p in config.get("history_sources", [])],
    )
    return pipeline, config


def demo(output):
    """Four synthetic task fixtures through every stage including snapshot retrieval."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ("topline", "continuous_normal", {}),
        ("profile", "profile_energy", {"cells": ["a", "b"]}),
        (
            "ranking-fixed",
            "ranking_list",
            {"ranking": {"length": 3, "items": ["A", "B", "C"]}},
        ),
        (
            "ranking-open",
            "ranking_list",
            {
                "ranking": {
                    "length": 3,
                    "project": "en.wikipedia.org",
                    "exclusions": {"titles": ["Main_Page"]},
                }
            },
        ),
    ]
    rows = []
    for label, kind, extra in specs:
        rid = "demo-" + label
        context = {
            "history": [
                {"date": "2026-09-17", "value": 40},
                {"date": "2026-09-18", "value": 41},
            ]
        }
        if kind == "profile_energy":
            context = {
                "history_by_cell": {
                    "a": [{"date": "2026-09-18", "value": 10}],
                    "b": [{"date": "2026-09-18", "value": 20}],
                }
            }
            response = {
                "cells": {"a": {"mean": 11, "sd": 1}, "b": {"mean": 21, "sd": 1}}
            }
        elif kind == "ranking_list":
            context = {
                "recent_weeks": [{"date": "2026-09-13", "items": ["A", "B", "C"]}]
            }
            response = (
                {"ranking": ["D", "B", "A"]}
                if label == "ranking-open"
                else {"scores": {"A": 1, "B": 3, "C": 2}, "ranking": ["B", "C", "A"]}
            )
        else:
            response = {"mean": 42, "sd": 1.5}
        envelope = {
            "schema_version": "ssa-agent-api-v2",
            "request_id": "fixture:" + rid,
            "round": {
                "round_id": rid,
                "board_id": {
                    "continuous_normal": "topline",
                    "profile_energy": "profile",
                    "ranking_list": "ranking",
                }[kind],
                "target_type": kind,
                "question": "Example approval attention forecast for September 27, 2026.",
                "unit": "points",
                "lock_at": "2026-09-21T00:00:00Z",
                "context": context,
                **extra,
            },
            "optional_crosstabs": [],
        }
        corpus = {
            "synthetic": True,
            "documents": [
                {
                    "url": "https://example.org/evidence",
                    "title": "Example approval attention historical forecast",
                    "content": "Example approval attention forecast historical measurement news developments. Synthetic test evidence only.",
                    "published_at": "2026-09-18T00:00:00Z",
                    "captured_at": "2026-09-18T01:00:00Z",
                    "candidate_items": ["D"],
                }
            ],
        }
        write(output / (label + "-envelope.json"), envelope)
        write(output / (label + "-corpus.json"), corpus)
        write(output / (label + "-responses.json"), {rid: response})
        config = {
            "pipeline": {"entrant": "demo", "mode": "snapshot"},
            "run_db": "runs.sqlite3",
            "research": {
                "provider": "snapshot",
                "corpus": label + "-corpus.json",
                "db": "research.sqlite3",
            },
            "models": [
                {
                    "kind": "replay",
                    "member_id": "fixture",
                    "responses": label + "-responses.json",
                }
            ],
            "task_metadata": {"forecast_for": "2026-09-27"},
        }
        path = output / (label + "-config.json")
        write(path, config)
        pipeline, _ = load_pipeline(path)
        result = pipeline.run_envelope(envelope, metadata=config["task_metadata"])
        write(output / (label + "-run.json"), result)
        rows.append({"task": label, "response": result["response"], "synthetic": True})
    return {"synthetic": True, "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description="MIT SSA unified forecasting harness")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--output-dir", default="data/demo")
    run = sub.add_parser("forecast")
    run.add_argument("--config", required=True)
    inp = run.add_mutually_exclusive_group(required=True)
    inp.add_argument("--envelope")
    inp.add_argument("--request")
    run.add_argument("--asof")
    run.add_argument("--output")
    run.add_argument("--full-run", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--keys")
    serve.add_argument("--no-verify", action="store_true")
    serve.add_argument("--enable-probe-fixtures", action="store_true")
    score = sub.add_parser("evaluate")
    score.add_argument("--records", required=True)
    score.add_argument("--output")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--cases", required=True)
    cal.add_argument("--validation-start", required=True)
    cal.add_argument("--test-start", required=True)
    cal.add_argument("--objective", choices=["mean_loss", "mean_skill"], default="mean_loss")
    cal.add_argument("--output")
    bt = sub.add_parser("backtest")
    bt.add_argument("--config", required=True)
    bt.add_argument("--cases", required=True)
    bt.add_argument("--output")
    source = sub.add_parser("import-source")
    source.add_argument(
        "--kind",
        choices=["series-csv", "wikipedia-json", "trends-csv", "civiqs", "yougov"],
        required=True,
    )
    source.add_argument("--input")
    source.add_argument("--manifest")
    source.add_argument("--output", required=True)
    source.add_argument("--cell-column")
    source.add_argument("--items", nargs="+")
    source.add_argument("--shares", action="store_true")
    source.add_argument("--tracker")
    source.add_argument("--filters", help="JSON object for Civiqs subgroup filters")
    source.add_argument("--user-agent")
    weekly = sub.add_parser("aggregate-wikipedia-week")
    weekly.add_argument("--packages", nargs="+", required=True)
    weekly.add_argument("--week-start", required=True)
    weekly.add_argument("--length", type=int, default=10)
    weekly.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.output_dir)
        elif args.command in ("forecast", "serve", "backtest"):
            pipeline, config = load_pipeline(args.config)
            if args.command == "serve":
                from .server import make_server

                keys = None
                if args.keys:
                    raw = read(args.keys)
                    keys = {r["key_id"]: r["public_key"] for r in raw["keys"]}
                server = make_server(
                    pipeline,
                    host=args.host,
                    port=args.port,
                    keys=keys,
                    verify=not args.no_verify,
                    metadata=config.get("task_metadata"),
                    metadata_by_round=config.get("task_metadata_by_round"),
                    enable_probe_fixtures=args.enable_probe_fixtures,
                )
                print(
                    f"Listening on {args.host}:{server.server_port}; /forecast and /health",
                    flush=True,
                )
                try:
                    server.serve_forever()
                finally:
                    server.server_close()
                return 0
            if args.command == "backtest":
                from .evaluation import walk_forward

                if pipeline.config.mode != "snapshot":
                    raise ValueError("backtesting requires snapshot mode")
                result = walk_forward(read(args.cases), pipeline.run)
            else:
                if args.envelope:
                    incoming = read(args.envelope)
                    metadata = {
                        **config.get("task_metadata", {}),
                        **config.get("task_metadata_by_round", {}).get(
                            incoming["round"]["round_id"], {}
                        ),
                    }
                    result = pipeline.run_envelope(
                        incoming, asof=args.asof, metadata=metadata
                    )
                else:
                    result = pipeline.run(ForecastRequest.from_dict(read(args.request)))
                if not args.full_run:
                    result = result["response"]
        elif args.command == "evaluate":
            from .evaluation import score_records

            result = score_records(read(args.records))
        elif args.command == "calibrate":
            from .evaluation import calibrate

            result = calibrate(
                read(args.cases),
                validation_start=args.validation_start,
                test_start=args.test_start,
                objective=args.objective,
            )
        elif args.command == "aggregate-wikipedia-week":
            from .sources import wikipedia_week

            result = wikipedia_week(
                [read(p) for p in args.packages],
                week_start=args.week_start,
                length=args.length,
            )
        else:
            from .sources import series_csv, wikipedia_snapshot, trends_csv

            if args.kind == "civiqs":
                from .source_adapters import fetch_source_package

                if not args.tracker:
                    raise ValueError("--tracker required for Civiqs")
                result = fetch_source_package(
                    "civiqs",
                    tracker=args.tracker,
                    filters=json.loads(args.filters) if args.filters else None,
                    output=args.output,
                    user_agent=args.user_agent
                    or "MIT-SSA-research/0.1 (contact project maintainer)",
                )
                args.output = None
            elif args.kind == "yougov":
                from .source_adapters import fetch_source_package

                if not args.tracker:
                    raise ValueError("--tracker required for YouGov")
                result = fetch_source_package(
                    "yougov",
                    tracker=args.tracker,
                    output=args.output,
                    user_agent=args.user_agent
                    or "MIT-SSA-research/0.1 (contact project maintainer)",
                )
                args.output = None
            else:
                if not args.input or not args.manifest:
                    raise ValueError(
                        "--input and --manifest are required for imported source kinds"
                    )
                manifest = read(args.manifest)
                if args.kind == "series-csv":
                    result = series_csv(
                        args.input, manifest, cell_column=args.cell_column
                    )
                elif args.kind == "wikipedia-json":
                    result = wikipedia_snapshot(args.input, manifest)
                else:
                    if not args.items:
                        raise ValueError("--items required for Trends")
                    result = trends_csv(
                        args.input, manifest, items=args.items, shares=args.shares
                    )
        if getattr(args, "output", None):
            write(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[{type(exc).__name__}] {exc}", file=sys.stderr)
        return 2
