"""Compare independent metric implementations against pinned official functions.

Downloads source into a temporary directory, evaluates pure metric functions,
and prints absolute differences. No official source is installed as our package.
"""

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from ssa_harness.evaluation import energy_normal, rbo_loss, kendall_loss


COMMIT = "e80c8b42c157dadc018441da2a9691b44b20affc"


def main():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/official-metric-golden.json"
    )
    data = json.loads(fixture.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("scoring", "ranking_round"):
            with urlopen(
                f"https://raw.githubusercontent.com/Social-Atoms/social-sim-arena/{COMMIT}/ssa/{name}.py",
                timeout=30,
            ) as r:
                (root / (name + ".py")).write_bytes(r.read())
        spec = importlib.util.spec_from_file_location(
            "official_metric_reference", root / "scoring.py"
        )
        score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(score)
        parsed = ast.parse((root / "ranking_round.py").read_text())
        names = {"rbo_similarity", "rbo_loss", "kendall_loss"}
        nodes = [
            n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        namespace = {}
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), "official-pure-metrics", "exec"
            ),
            namespace,
        )
        rows = []
        for case in data["cases"]:
            kind = case["kind"]
            if kind == "energy":
                cells, truth, order = case["cells"], case["truth"], case["order"]
                reference = score.energy_score(
                    score.cell_samples([cells[c] for c in order]),
                    [truth[c] for c in order],
                )
                actual = energy_normal(cells, truth, order)
            elif kind == "rbo":
                reference = namespace["rbo_loss"](
                    case["prediction"], case["truth"], case["p"]
                )
                actual = rbo_loss(case["prediction"], case["truth"], case["p"])
            else:
                reference = namespace["kendall_loss"](
                    case["prediction"], case["truth"]
                )[0]
                actual = kendall_loss(case["prediction"], case["truth"])
            delta = abs(actual - reference)
            if delta > 1e-7 or abs(reference - case["expected"]) > 1e-12:
                raise AssertionError("metric mismatch")
            rows.append(
                {
                    "metric": kind,
                    "official": reference,
                    "local": actual,
                    "absolute_difference": delta,
                }
            )
        print(
            json.dumps(
                {"official_commit": COMMIT, "tolerance": 1e-7, "checks": rows}, indent=2
            )
        )


if __name__ == "__main__":
    main()
