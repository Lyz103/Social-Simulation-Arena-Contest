"""Command line interface with an offline demo and JSON integration."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from . import (
    ArchiveStore,
    EngineConfig,
    ResearchBundle,
    ResearchEngine,
    ResearchError,
    ResearchRequest,
    SnapshotRetriever,
    TavilyRetriever,
)

EXAMPLES = Path(__file__).parent / "examples"


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立 Harness 检索模块")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="离线运行虚构资料示例，不产生 API 费用")
    demo.add_argument(
        "--kind", choices=("topline", "profile", "ranking"), default="topline"
    )
    demo.add_argument("--db", default="data/demo.sqlite3")
    run = commands.add_parser("run", help="从 JSON 请求创建或回放 bundle")
    run.add_argument("--request", required=True)
    run.add_argument("--provider", choices=("snapshot", "tavily"), required=True)
    run.add_argument("--corpus", help="snapshot provider 必填")
    run.add_argument("--config", help="可选 EngineConfig JSON 文件")
    run.add_argument("--entrant", default="default")
    run.add_argument("--db", default="data/research.sqlite3")
    run.add_argument("--output", help="导出 UTF-8 bundle JSON；拒绝覆盖不同内容")
    verify = commands.add_parser("verify", help="校验导出的 bundle SHA-256")
    verify.add_argument("bundle")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            bundle = ResearchBundle.from_dict(_read(args.bundle))
            print("校验通过：" + bundle.bundle_hash)
            return 0
        config = EngineConfig()
        if args.command == "demo":
            request = ResearchRequest.from_dict(_read(EXAMPLES / f"{args.kind}.json"))
            retriever = SnapshotRetriever(EXAMPLES / "corpus.json")
            entrant, output = "demo", None
        else:
            request = ResearchRequest.from_dict(_read(args.request))
            if args.config:
                config = EngineConfig(**_read(args.config))
            if args.provider == "snapshot" and not args.corpus:
                parser.error("snapshot provider 需要 --corpus")
            retriever = (
                SnapshotRetriever(args.corpus)
                if args.provider == "snapshot"
                else TavilyRetriever()
            )
            entrant, output = args.entrant, args.output
        bundle = ResearchEngine(
            retriever=retriever, store=ArchiveStore(args.db), config=config
        ).retrieve(request, entrant=entrant)
        if output:
            dest = Path(output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                existing = ResearchBundle.from_dict(_read(dest))
                if existing.bundle_hash != bundle.bundle_hash:
                    raise ResearchError("导出路径已存在不同 bundle，请更换文件名")
            else:
                # Exclusive create protects an existing user artifact.
                with dest.open("x", encoding="utf-8") as stream:
                    json.dump(bundle.to_dict(), stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
        if args.command == "demo":
            print(bundle.rendered_text)
            print("\n状态：", bundle.status, "证据数：", len(bundle.documents))
            for warning in bundle.warnings:
                print("说明：", warning)
            print("bundle_hash：", bundle.bundle_hash)
        else:
            print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except (ResearchError, OSError, sqlite3.Error, ValueError, TypeError) as exc:
        print(f"检索失败 [{type(exc).__name__}]：{exc}", file=sys.stderr)
        return 2


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
