import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from test_integration import request, envelope
from ssa_harness.archive import RunArchive
from ssa_harness.contracts import ForecastRequest
from ssa_harness.evaluation import (
    crps_normal,
    energy_normal,
    rbo_loss,
    kendall_loss,
    score_prediction,
    calibrate,
    walk_forward,
)
from ssa_harness.history import normalize_history
from ssa_harness.llm import ReplayClient
from ssa_harness.pipeline import ForecastPipeline, PipelineConfig
from ssa_harness.server import make_server, verify_signature
from ssa_harness.sources import (
    series_csv,
    trends_csv,
    wikipedia_snapshot,
    fetch_wikipedia_daily,
)


def calibration_cases():
    cases = []
    for i, day in enumerate(
        (
            "2026-08-01",
            "2026-08-08",
            "2026-08-15",
            "2026-08-22",
            "2026-09-01",
            "2026-09-08",
        )
    ):
        from datetime import datetime, timedelta, timezone

        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        iso = lambda dt: dt.isoformat().replace("+00:00", "Z")
        r = request(
            round_id=f"cal-{i}",
            asof=iso(start),
            deadline=iso(start + timedelta(days=1)),
            history=(),
        )
        cases.append(
            {
                "request": r.to_dict(),
                "truth": 42 + i,
                "truth_available_at": iso(start + timedelta(days=2)),
                "members": [
                    {
                        "member_id": "model",
                        "member_type": "llm",
                        "prediction": {"mean": 42 + i, "sd": 1},
                    },
                    {
                        "member_id": "persistence",
                        "member_type": "persistence",
                        "prediction": {"mean": 40 + i, "sd": 1},
                    },
                    {
                        "member_id": "recent_mean",
                        "member_type": "recent_mean",
                        "prediction": {"mean": 39 + i, "sd": 1},
                    },
                ],
            }
        )
    return cases


class EvaluationTests(unittest.TestCase):
    def test_public_topline_example(self):
        self.assertAlmostEqual(crps_normal(34.5, 1, 35), 0.331403531255, places=11)
        self.assertAlmostEqual(crps_normal(33, 1.5, 35), 1.280900969803, places=11)

    def test_energy_one_dimensional_matches_empirical_crps(self):
        from ssa_harness.evaluation import _profile_grid

        values = _profile_grid(0)
        expected = sum(abs(x - 0.2) for x in values) / 400 - sum(
            abs(x - y) for i, x in enumerate(values) for y in values[i + 1 :]
        ) / (400**2)
        self.assertAlmostEqual(
            energy_normal({"x": {"mean": 0, "sd": 1}}, {"x": 0.2}, ["x"]),
            expected,
            places=12,
        )

    def test_ranking_metrics_extremes_and_known_kendall(self):
        self.assertEqual(rbo_loss(["A", "B"], ["A", "B"]), 0)
        self.assertEqual(rbo_loss(["A", "B"], ["C", "D"]), 1)
        self.assertEqual(kendall_loss(["A", "B", "C"], ["C", "B", "A"]), 1)
        self.assertAlmostEqual(kendall_loss(["A", "C", "B"], ["A", "B", "C"]), 1 / 3)

    def test_calibration_does_not_select_on_test_truth(self):
        cases = calibration_cases()
        first = calibrate(
            cases,
            validation_start="2026-08-01T00:00:00Z",
            test_start="2026-09-01T00:00:00Z",
        )
        changed = copy.deepcopy(cases)
        changed[-1]["truth"] = -1000
        second = calibrate(
            changed,
            validation_start="2026-08-01T00:00:00Z",
            test_start="2026-09-01T00:00:00Z",
        )
        self.assertEqual(first["selected"], second["selected"])
        self.assertNotEqual(first["test"], second["test"])

    def test_validation_truth_must_be_available_before_test(self):
        cases = calibration_cases()
        cases[0]["truth_available_at"] = "2026-09-10T00:00:00Z"
        with self.assertRaises(ValueError):
            calibrate(
                cases,
                validation_start="2026-08-01T00:00:00Z",
                test_start="2026-09-01T00:00:00Z",
            )

    def test_walk_forward_never_passes_truth_to_predictor(self):
        seen = []

        def predict(r):
            seen.append(r)
            self.assertIsInstance(r, ForecastRequest)
            self.assertNotIn("truth", r.to_dict())
            return {"mean": 42, "sd": 1}

        cases = calibration_cases()
        out = walk_forward(cases, predict)
        self.assertEqual(len(seen), 6)
        self.assertEqual(out["evaluation"]["by_metric"]["crps"]["count"], 6)

    def test_official_numeric_golden(self):
        path = Path(__file__).parent / "fixtures" / "official-metric-golden.json"
        data = json.loads(path.read_text())
        for case in data["cases"]:
            if case["kind"] == "energy":
                got = energy_normal(case["cells"], case["truth"], case["order"])
            elif case["kind"] == "rbo":
                got = rbo_loss(case["prediction"], case["truth"], case["p"])
            else:
                got = kendall_loss(case["prediction"], case["truth"])
            self.assertAlmostEqual(got, case["expected"], delta=1e-7)


class SourceTests(unittest.TestCase):
    def test_source_cannot_embed_future_release_in_old_capture(self):
        p = self.path / "future.csv"
        p.write_text("date,value,available_at\n2026-09-18,40,2026-09-25T00:00:00Z\n")
        with self.assertRaises(ValueError):
            series_csv(p, self.meta)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.meta = {
            "source_name": "Synthetic structured fixture",
            "source_url": "https://example.org/data",
            "captured_at": "2026-09-19T10:00:00Z",
            "unit": "points",
            "population": "test population",
        }

    def test_series_and_profile_csv(self):
        p = self.path / "poll.csv"
        p.write_text(
            "date,value,cell,available_at\n2026-09-18,40,a,2026-09-19T09:00:00Z\n2026-09-18,50,b,2026-09-19T09:00:00Z\n"
        )
        out = series_csv(p, self.meta, cell_column="cell")
        r = request(
            target_type="profile_energy",
            cells=("a", "b"),
            history=tuple(out["history"]),
        )
        self.assertEqual(len(normalize_history(r).rows), 2)
        self.assertTrue(out["manifest"]["content_hash"].startswith("sha256:"))

    def test_manifest_tampering_and_missing_availability(self):
        p = self.path / "poll.csv"
        p.write_text("date,value\n2026-09-18,40\n")
        with self.assertRaises(ValueError):
            series_csv(p, self.meta)
        with self.assertRaises(ValueError):
            series_csv(p, {**self.meta, "content_hash": "wrong"})

    def test_current_capture_does_not_become_old_snapshot(self):
        p = self.path / "poll.csv"
        p.write_text("date,value,available_at\n2026-09-18,40,2026-09-19T00:00:00Z\n")
        out = series_csv(p, {**self.meta, "captured_at": "2026-09-24T00:00:00Z"})
        self.assertEqual(
            normalize_history(request(history=tuple(out["history"]))).rows, ()
        )

    def test_trends_joint_basket_and_shares(self):
        p = self.path / "trends.csv"
        p.write_text("date,A,B,available_at\n2026-09-18,20,80,2026-09-19T00:00:00Z\n")
        with self.assertRaises(ValueError):
            trends_csv(p, self.meta, items=["A", "B"])
        out = trends_csv(
            p, {**self.meta, "joint_basket": True}, items=["A", "B"], shares=True
        )
        self.assertEqual(out["history"][0]["value"], {"A": 20, "B": 80})

    def test_wikipedia_json_filters_namespaces(self):
        p = self.path / "wiki.json"
        p.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "year": "2026",
                            "month": "09",
                            "day": "18",
                            "articles": [
                                {"article": "Main_Page", "views": 999},
                                {"article": "A B", "views": 20},
                                {"article": "C", "views": 10},
                            ],
                        }
                    ]
                }
            )
        )
        out = wikipedia_snapshot(p, self.meta)
        self.assertEqual(out["history"][0]["value"], ["A_B", "C"])

    def test_weekly_aggregation_requires_seven_days(self):
        from ssa_harness.sources import wikipedia_week

        packages = []
        for day in range(7, 14):
            p = self.path / f"wiki-{day}.json"
            p.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "year": "2026",
                                "month": "09",
                                "day": f"{day:02}",
                                "articles": [
                                    {"article": "A", "views": 20},
                                    {"article": "B", "views": 10},
                                ],
                            }
                        ]
                    }
                )
            )
            packages.append(wikipedia_snapshot(p, self.meta))
        with self.assertRaises(ValueError):
            wikipedia_week(packages[:6], week_start="2026-09-07", length=2)
        out = wikipedia_week(packages, week_start="2026-09-07", length=2)
        self.assertEqual(out["history"][0]["views"], {"A": 140, "B": 70})
        self.assertEqual(out["manifest"]["frequency"], "weekly")

    def test_source_merge_rejects_other_population(self):
        from ssa_harness.sources import merge_source_history, StructuredRetriever

        p = self.path / "poll.csv"
        p.write_text("date,value,available_at\n2026-09-18,40,2026-09-19T00:00:00Z\n")
        package = series_csv(p, {**self.meta, "metric": "approval"})
        r = request(
            source_name=self.meta["source_name"],
            population=self.meta["population"],
            unit="points",
            metric="approval",
            history=(),
        )
        merged = merge_source_history(r, [package])
        self.assertEqual(normalize_history(merged).rows[0]["value"], 40)
        from dataclasses import replace

        with self.assertRaises(ValueError):
            merge_source_history(replace(r, population="different"), [package])
        pp = self.path / "package.json"
        pp.write_text(json.dumps(package))
        retriever = StructuredRetriever([pp])
        from ssa_harness.retrieval import SearchQuery

        results = retriever.search(
            SearchQuery("approval points", intent="history", topic="general"),
            r.to_research_request(),
            max_results=3,
            lookback_days=14,
            timeout=1,
        )
        self.assertEqual(len(results), 1)


class ServerTests(unittest.TestCase):
    def setUp(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.key = Ed25519PrivateKey.generate()
        self.keys = {
            "local-test": base64.b64encode(
                self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            ).decode()
        }
        response = {"cells": {"a": {"mean": 10, "sd": 1}, "b": {"mean": 20, "sd": 1}}}
        self.client = ReplayClient({"contract-001": response})
        pipeline = ForecastPipeline(
            archive=RunArchive(Path(self.tmp.name) / "run.sqlite3"),
            clients={"fixture": self.client},
            config=PipelineConfig(research_policy="disabled"),
        )
        self.server = make_server(
            pipeline, port=0, keys=self.keys, enable_probe_fixtures=True
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/forecast"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def call(self, value, bad=False):
        raw = json.dumps(value).encode()
        ts = str(int(time.time()))
        sig = self.key.sign(ts.encode() + b"." + raw)
        if bad:
            sig = bytes([sig[0] ^ 1]) + sig[1:]
        req = Request(
            self.url,
            data=raw,
            headers={
                "Content-Type": "application/json",
                "X-SSA-Key-Id": "local-test",
                "X-SSA-Timestamp": ts,
                "X-SSA-Signature": base64.b64encode(sig).decode(),
            },
        )
        with urlopen(req, timeout=10) as r:
            return json.load(r)

    def test_real_pipeline_http_replay_and_cors(self):
        first = self.call(envelope())
        second = self.call(envelope())
        self.assertEqual(first, second)
        self.assertEqual(self.client.calls, 1)
        with urlopen(Request(self.url, method="OPTIONS")) as r:
            self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")

    def test_invalid_signature_rejected(self):
        with self.assertRaises(HTTPError) as cm:
            self.call(envelope(), bad=True)
        self.assertEqual(cm.exception.code, 401)
        cm.exception.close()

    def test_changed_request_returns_conflict(self):
        self.call(envelope())
        changed = envelope()
        changed["round"]["question"] = "changed"
        with self.assertRaises(HTTPError) as cm:
            self.call(changed)
        self.assertEqual(cm.exception.code, 409)
        cm.exception.close()

    def test_official_probe_fixture_does_not_call_model(self):
        e = envelope()
        e["round"] = {
            "round_id": "ssa-contract-test",
            "board_id": "topline",
            "target_type": "continuous_normal",
            "question": "Non-scored contract test: return a normal forecast centered on 50.",
            "unit": "points",
            "lock_at": "2099-01-01T00:00:00Z",
            "context": {"persistence": 50},
        }
        self.assertEqual(self.call(e)["forecast"], {"mean": 50, "sd": 5})
        self.assertEqual(self.client.calls, 0)


if __name__ == "__main__":
    unittest.main()
