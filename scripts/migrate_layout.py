"""One-time, non-destructive migration; originals stay in archive/legacy/."""

from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
archive = root / "archive" / "legacy"
package = root / "src" / "ssa_harness"
if archive.exists():
    raise SystemExit("Migration already performed; refusing to repeat it")
for name in ("Arena_competition", "harness_retrieval"):
    source = (root / name).resolve()
    if source.parent != root or not source.is_dir():
        raise SystemExit(f"Unexpected migration source: {source}")
package.mkdir(parents=True)
shutil.copytree(
    root / "harness_retrieval/retrieval",
    package / "retrieval",
    ignore=shutil.ignore_patterns("__pycache__"),
)
shutil.copytree(
    root / "Arena_competition/ssa/forecast",
    package / "prompt",
    ignore=shutil.ignore_patterns("__pycache__"),
)
shutil.copytree(
    root / "Arena_competition/ssa/ensemble",
    package / "ensemble",
    ignore=shutil.ignore_patterns("__pycache__"),
)
tests = root / "tests"
tests.mkdir()
for old in (root / "harness_retrieval/tests", root / "Arena_competition/tests"):
    for file in old.glob("test_*.py"):
        text = file.read_text(encoding="utf-8")
        text = text.replace("from retrieval", "from ssa_harness.retrieval").replace(
            '"retrieval.providers.', '"ssa_harness.retrieval.providers.'
        )
        text = text.replace("ssa.forecast", "ssa_harness.prompt").replace(
            "ssa.ensemble", "ssa_harness.ensemble"
        )
        text = text.replace(
            'Path(__file__).resolve().parents[1] / "retrieval" / "examples"',
            'Path(__import__("ssa_harness.retrieval", fromlist=["__file__"]).__file__).parent / "examples"',
        )
        (tests / file.name).write_text(text, encoding="utf-8")
for file in (package / "ensemble").glob("*.py"):
    file.write_text(
        file.read_text(encoding="utf-8").replace("ssa.forecast", "ssa_harness.prompt"),
        encoding="utf-8",
    )
docs = root / "docs" / "design"
docs.mkdir(parents=True)
for file in (root / "Arena_competition").glob("*.md"):
    shutil.copy2(file, docs / file.name)
archive.mkdir(parents=True)
for name in ("Arena_competition", "harness_retrieval"):
    source = (root / name).resolve()
    destination = (archive / name).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root):
        raise SystemExit("Migration escaped workspace")
    shutil.move(str(source), str(destination))
print(
    "Migrated sources and tests; preserved both original projects under archive/legacy"
)
