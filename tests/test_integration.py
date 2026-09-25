import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssa_harness.archive import RunArchive
from ssa_harness.cli import demo, load_pipeline
from ssa_harness.context import ContextBuilder, ContextConfig
from ssa_harness.contracts import (
    ForecastRequest,
    ContractError,
    response_envelope,
    validate_schema,
)
from ssa_harness.ensemble import BaselineFactory, EnsembleEngine, ForecastMember
from ssa_harness.history import normalize_history, normalized_request, HistoryError
from ssa_harness.llm import ChatCompletionsClient, ReplayClient, ModelError
from ssa_harness.pipeline import ForecastPipeline, PipelineConfig
from ssa_harness.prompt import (
    ForecastPromptCompiler,
    PredictionValidator,
    InvalidModelOutput,
)
from ssa_harness.ranking import CandidateBuilder
from ssa_harness.retrieval.errors import ArchiveConflict, ArchiveCorruption


def request(**changes):
    values = dict(
        round_id="audit-round",
        question="Forecast next approval",
        target_type="continuous_normal",
        unit="%",
        asof="2026-09-20T00:00:00Z",
        deadline="2026-09-21T00:00:00Z",
        history=(
            {"date": "2026-09-18", "available_at": "2026-09-19T00:00:00Z", "value": 40},
            {"date": "2026-09-19", "available_at": "2026-09-19T12:00:00Z", "value": 41},
        ),
    )
    values.update(changes)
    return ForecastRequest(**values)


def envelope(kind="profile_energy"):
    ctx = {
        "history_by_cell": {
            "a": [{"date": "2026-09-18", "value": 10}],
            "b": [{"date": "2026-09-18", "value": 20}],
        }
    }
    extra = {"cells": ["a", "b"]}
    if kind == "ranking_list":
        ctx = {"recent_weeks": [{"date": "2026-09-13", "items": ["A", "B", "C"]}]}
        extra = {"ranking": {"length": 3, "project": "en.wikipedia.org"}}
    return {
        "schema_version": "ssa-agent-api-v2",
        "request_id": "test:contract-001",
        "round": {
            "round_id": "contract-001",
            "board_id": "profile" if kind == "profile_energy" else "ranking",
            "target_type": kind,
            "question": "Forecast the next observation",
            "unit": "points",
            "lock_at": "2026-09-21T00:00:00Z",
            "context": ctx,
            **extra,
        },
        "optional_crosstabs": [],
    }


class ContractHistoryTests(unittest.TestCase):
    def test_official_profile_history_maps_to_baseline(self):
        r = ForecastRequest.from_envelope(envelope())
        self.assertEqual(r.asof, "2026-09-20T00:00:00Z")
        self.assertEqual(
            BaselineFactory().build(r)[0].prediction["cells"]["b"]["mean"], 20
        )

    def test_official_ranking_items_history(self):
        r = ForecastRequest.from_envelope(envelope("ranking_list"))
        self.assertTrue(r.is_open_ranking)
        self.assertEqual(
            BaselineFactory().build(r)[0].prediction, {"ranking": ["A", "B", "C"]}
        )

    def test_earlier_cutoff_excludes_unproven_official_history(self):
        r = ForecastRequest.from_envelope(envelope(), asof="2026-09-19T00:00:00Z")
        self.assertEqual(normalize_history(r).rows, ())

    def test_future_and_unsorted_history(self):
        r = request(
            history=tuple(reversed(request().history))
            + (
                {
                    "date": "2026-09-19",
                    "available_at": "2026-09-25T00:00:00Z",
                    "value": 99,
                },
            )
        )
        clean, h = normalized_request(r)
        self.assertEqual([x["value"] for x in h.rows], [40, 41])
        self.assertEqual(BaselineFactory().build(r)[0].prediction["mean"], 41)
        self.assertIn("excluded", h.warnings[0])

    def test_missing_availability_and_conflicting_duplicates_fail(self):
        with self.assertRaises(HistoryError):
            normalize_history(request(history=({"date": "2026-09-18", "value": 40},)))
        row = request().history[0]
        with self.assertRaises(HistoryError):
            normalize_history(request(history=(row, {**row, "value": 99})))

    def test_revision_uses_latest_available_value(self):
        first = request().history[0]
        newer = {**first, "available_at": "2026-09-19T12:00:00Z", "value": 42}
        self.assertEqual(
            normalize_history(request(history=(newer, first))).rows[0]["value"], 42
        )

    def test_profile_and_open_response_pass_official_schema(self):
        r = ForecastRequest.from_envelope(envelope())
        result = response_envelope(
            r, {"cells": {"a": {"mean": 10, "sd": 1}, "b": {"mean": 20, "sd": 1}}}
        )
        self.assertIn("profile", result["forecast"])
        r = ForecastRequest.from_envelope(envelope("ranking_list"))
        self.assertEqual(
            response_envelope(r, {"ranking": ["Alpha Beta", "B", "C"]})["forecast"][
                "ranking"
            ][0],
            "Alpha_Beta",
        )

    def test_invalid_official_input_fails(self):
        e = envelope()
        e["round"]["unknown"] = True
        with self.assertRaises(ContractError):
            ForecastRequest.from_envelope(e)


class ContextPromptTests(unittest.TestCase):
    def bundle(self, req, documents, mode="snapshot"):
        from ssa_harness.retrieval import ResearchBundle

        return ResearchBundle(
            schema_version="research-bundle-v1",
            round_id=req.round_id,
            entrant="test",
            request_hash="test",
            pipeline_hash="test",
            request=req.to_research_request(mode=mode).to_dict(),
            params={},
            created_at=req.asof,
            status="complete",
            temporal_assurance=(
                "snapshot_capture_asserted"
                if mode == "snapshot"
                else "publication_metadata_only"
            ),
            queries=[],
            documents=documents,
            exclusions=[],
            retrieval_log=[],
            rendered_text="test",
            warnings=[],
            stats={},
        ).seal()

    def test_context_rechecks_document_time_and_preserves_conflicts(self):
        req = request()
        base = {
            "title": "Measurement",
            "url": "https://e.org/a",
            "published_at": "2026-09-18T00:00:00Z",
            "captured_at": "2026-09-19T00:00:00Z",
            "modified_at": None,
            "content": "Observed evidence",
            "evidence_id": "E1",
            "score": 1,
            "matches": [{"intent": "measurement"}],
        }
        documents = [
            base,
            {
                **base,
                "url": "https://other.org/b",
                "evidence_id": "E2",
                "stance": "down",
            },
            {
                **base,
                "url": "https://e.org/future",
                "captured_at": "2026-09-22T00:00:00Z",
                "evidence_id": "E3",
            },
        ]
        context = ContextBuilder().build(req, self.bundle(req, documents))
        self.assertEqual(context.sections["CONFLICTING EVIDENCE"][0]["id"], "E2")
        self.assertEqual(context.sections["RELATED INDICATORS"][0]["id"], "E1")
        self.assertIn("capture_after_cutoff", " ".join(context.warnings))
        self.assertNotIn('"id": "E3"', context.rendered_text)

    def test_strict_context_refuses_live_metadata_only(self):
        req = request()
        with self.assertRaises(ValueError):
            ContextBuilder().build(req, self.bundle(req, [], mode="live"))

    def test_profile_context_with_sixteen_cells_fits_default_budget(self):
        cells = tuple(f"cell-{i}" for i in range(16))
        history = tuple(
            {
                "date": f"2026-09-{day:02}",
                "available_at": f"2026-09-{day:02}T23:59:59Z",
                "value": {c: 20 + i for i, c in enumerate(cells)},
            }
            for day in (16, 17, 18)
        )
        req = request(target_type="profile_energy", cells=cells, history=history)
        context = ContextBuilder().build(req)
        self.assertLessEqual(len(context.rendered_text), 14000)
        self.assertEqual(len(context.history_summary), 16)

    def test_history_rendered_and_hash_verified(self):
        r, _ = normalized_request(request())
        c = ContextBuilder().build(r)
        c.verify()
        p = ForecastPromptCompiler().compile(r, c)
        self.assertIn('"latest": 41.0', p.user_message)
        self.assertEqual(c, ContextBuilder().build(r))
        self.assertNotIn("Forecast target time/window:\n2026-09-21", p.user_message)

    def test_missing_identity_wrong_hash_and_tampering_fail(self):
        r = request()
        c = ContextBuilder().build(r)
        with self.assertRaises(ValueError):
            ForecastPromptCompiler().compile(r, replace(c, rendered_text="bad"))
        with self.assertRaises(Exception):
            ForecastPromptCompiler().compile(r, {"rendered_text": "evidence"})
        with self.assertRaises(Exception):
            ForecastPromptCompiler().compile(replace(r, question="different"), c)

    def test_mandatory_budget_fail_closed(self):
        with self.assertRaises(ValueError):
            ContextBuilder(ContextConfig(max_chars=30)).build(request())

    def test_open_candidates_are_suggestions(self):
        r = ForecastRequest.from_envelope(envelope("ranking_list"))
        r, _ = normalized_request(r)
        pool = CandidateBuilder().build(
            r,
            [
                {
                    "candidate_items": ["D", "Main_Page"],
                    "url": "https://e.org",
                    "evidence_id": "E1",
                }
            ],
        )
        self.assertIn("D", pool.items)
        self.assertNotIn("Main_Page", pool.items)
        r = replace(r, candidates=pool.items)
        p = ForecastPromptCompiler().compile(r, ContextBuilder().build(r))
        self.assertNotIn("scores", p.output_schema["required"])
        self.assertEqual(
            PredictionValidator().parse({"ranking": ["New article", "D", "B"]}, r)[
                "ranking"
            ][0],
            "New_article",
        )

    def test_canonical_duplicates_and_exclusions_rejected(self):
        r = ForecastRequest.from_envelope(envelope("ranking_list"))
        for rows in (
            ["A B", "A_B", "C"],
            ["Main_Page", "B", "C"],
            ["Special:Search", "B", "C"],
        ):
            with self.assertRaises(InvalidModelOutput):
                PredictionValidator().parse({"ranking": rows}, r)


class EnsembleRegressionTests(unittest.TestCase):
    def test_stable_variance_and_large_weights(self):
        r = request(history=())
        engine = EnsembleEngine()
        self.assertEqual(
            engine.combine(r, [{"mean": 1e8, "sd": 1}], baselines=[]).prediction["sd"],
            1,
        )
        members = [
            ForecastMember("a", "llm", {"mean": 42, "sd": 1}, 1e308),
            ForecastMember("b", "llm", {"mean": 42, "sd": 1}, 1e308),
        ]
        final = engine.combine(r, members, baselines=[])
        self.assertEqual(final.prediction["mean"], 42)
        self.assertAlmostEqual(sum(final.weights.values()), 1)

    def test_explicit_drop_policy_keeps_good_member_and_marks_degraded(self):
        final = EnsembleEngine(failure_policy="drop_invalid").combine(
            request(history=()), [{"mean": 42, "sd": 1}, {"mean": 43, "sd": 0}]
        )
        self.assertEqual(final.prediction["mean"], 42)
        self.assertTrue(final.degraded)

    def test_strict_policy_and_all_invalid_fail(self):
        with self.assertRaises(InvalidModelOutput):
            EnsembleEngine().combine(request(), [{"mean": 43, "sd": 0}])
        with self.assertRaises(Exception):
            EnsembleEngine(failure_policy="drop_invalid").combine(
                request(history=()), [{"mean": 43, "sd": 0}]
            )

    def test_open_list_ensemble_merges_different_items(self):
        r = ForecastRequest.from_envelope(envelope("ranking_list"))
        final = EnsembleEngine().combine(r, [{"ranking": ["D", "B", "A"]}])
        self.assertEqual(len(final.prediction["ranking"]), 3)
        self.assertIn("D", final.prediction["ranking"])

    def test_fixed_tie_rule_is_consistent(self):
        r = request(
            target_type="ranking_list",
            history=(),
            candidates=("A", "B"),
            ranking_spec={"length": 2, "items": ["A", "B"]},
        )
        final = EnsembleEngine(weights={"llm": 0.5, "persistence": 0.5}).combine(
            r,
            [{"scores": {"A": 1, "B": 0}, "ranking": ["A", "B"]}],
            [
                ForecastMember(
                    "persistence",
                    "persistence",
                    {"scores": {"A": 0, "B": 1}, "ranking": ["B", "A"]},
                )
            ],
        )
        self.assertEqual(final.prediction["ranking"], ["A", "B"])

    def test_large_opposite_ranking_scores_are_finite(self):
        r = request(
            target_type="ranking_list",
            history=(),
            candidates=("A", "B"),
            ranking_spec={"length": 2, "items": ["A", "B"]},
        )
        final = EnsembleEngine().combine(
            r,
            [{"scores": {"A": 1e308, "B": -1e308}, "ranking": ["A", "B"]}],
            baselines=[],
        )
        self.assertEqual(final.prediction["scores"], {"A": 1.0, "B": 0.0})

    def test_explicit_model_id_preserved(self):
        final = EnsembleEngine().combine(
            request(),
            [
                {"model_id": "llm-1", "prediction": {"mean": 42, "sd": 1}},
                {"model_id": "another", "prediction": {"mean": 41, "sd": 1}},
            ],
        )
        self.assertIn("llm-1", final.weights)
        self.assertIn("another", final.weights)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RunArchive(Path(self.tmp.name) / "runs.sqlite3")

    def pipeline(self, client=None, **config):
        return ForecastPipeline(
            archive=self.archive,
            clients={
                "fixture": client
                or ReplayClient({"audit-round": {"mean": 42, "sd": 1}})
            },
            config=PipelineConfig(research_policy="disabled", **config),
        )

    def test_pipeline_idempotent_without_second_model_call(self):
        client = ReplayClient({"audit-round": {"mean": 42, "sd": 1}})
        p = self.pipeline(client)
        first = p.run(request())
        second = p.run(request())
        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)
        self.assertTrue(
            any(
                e["stage"] == "model_raw"
                for e in self.archive.events(p.config.entrant, "audit-round")
            )
        )
        with self.assertRaises(ArchiveConflict):
            p.run(replace(request(), question="changed"))

    def test_full_run_tampering_detected(self):
        p = self.pipeline()
        p.run(request())
        with self.archive._db() as db:
            db.execute("UPDATE forecast_runs SET payload='{}'")
        with self.assertRaises(ArchiveCorruption):
            p.run(request())

    def test_failure_raw_preserved_and_fallback_explicit(self):
        bad = ReplayClient({"audit-round": "not json"})
        with self.assertRaises(ModelError):
            self.pipeline(bad).run(request())
        events = self.archive.events("local-experiment", "audit-round")
        self.assertEqual(sum(e["stage"] == "model_raw" for e in events), 2)
        final = self.pipeline(
            bad, entrant="fallback", allow_baseline_fallback=True
        ).run(request())
        self.assertTrue(final["final"]["degraded"])

    def test_model_client_payload_and_secret_handling(self):
        seen = []

        def transport(req, timeout):
            seen.append(json.loads(req.data))
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"{\\"mean\\":42,\\"sd\\":1}"},"finish_reason":"stop"}],"usage":{"total_tokens":20}}'
            )

        client = ChatCompletionsClient(
            model="test",
            endpoint="https://provider.example/v1/chat/completions",
            transport=transport,
        )
        r = request()
        compiled = ForecastPromptCompiler().compile(r, ContextBuilder().build(r))
        with patch.dict("os.environ", {"SSA_LLM_API_KEY": "secret-value"}):
            reply = client.generate(compiled)
        self.assertEqual(reply.usage["total_tokens"], 20)
        self.assertNotIn("secret-value", json.dumps(client.identity()))
        self.assertEqual(seen[0]["response_format"], {"type": "json_object"})

    def test_real_client_retries_only_transient_errors(self):
        from urllib.error import HTTPError

        calls = []

        def transport(req, timeout):
            calls.append(req)
            if len(calls) == 1:
                raise HTTPError(req.full_url, 429, "secret", {}, None)
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"{\\"mean\\":42,\\"sd\\":1}"}}]}'
            )

        r = request()
        compiled = ForecastPromptCompiler().compile(r, ContextBuilder().build(r))
        client = ChatCompletionsClient(
            model="test",
            endpoint="https://provider.example/v1/chat/completions",
            transport=transport,
            sleep=lambda x: None,
        )
        with patch.dict("os.environ", {"SSA_LLM_API_KEY": "secret"}):
            client.generate(compiled)
        self.assertEqual(len(calls), 2)

        def denied(req, timeout):
            raise HTTPError(req.full_url, 401, "secret", {}, None)

        client.transport = denied
        with patch.dict("os.environ", {"SSA_LLM_API_KEY": "secret"}):
            with self.assertRaises(ModelError) as cm:
                client.generate(compiled)
        self.assertNotIn("secret", str(cm.exception))

    def test_repair_result_archived_and_can_succeed(self):
        from ssa_harness.llm import ModelReply

        class Repairing(ReplayClient):
            def generate(self, compiled, *, timeout=None, repair=None):
                self.calls += 1
                return ModelReply(
                    '{"mean":42,"sd":1}' if repair else "invalid", {}, "repair-test"
                )

        client = Repairing({})
        p = self.pipeline(client)
        result = p.run(request())
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(result["model_runs"]), 2)
        self.assertFalse(result["final"]["degraded"])

    def test_successful_member_reused_after_interrupted_run(self):
        class InterruptOnce(EnsembleEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def combine(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("simulated interruption")
                return super().combine(*args, **kwargs)

        client = ReplayClient({"audit-round": {"mean": 42, "sd": 1}})
        p = self.pipeline(client)
        p.ensemble = InterruptOnce()
        with self.assertRaises(RuntimeError):
            p.run(request())
        result = p.run(request())
        self.assertEqual(client.calls, 1)
        self.assertTrue(result["response"])

    def test_expired_live_request_fails_before_model(self):
        class RealStub(ReplayClient):
            synthetic = False

        client = RealStub({"audit-round": {"mean": 42, "sd": 1}})
        p = self.pipeline(client, mode="live")
        with self.assertRaises(TimeoutError):
            p.run(request())
        self.assertEqual(client.calls, 0)

    def test_live_refuses_synthetic_models(self):
        with self.assertRaises(ValueError):
            self.pipeline(mode="live")

    def test_four_complete_demos_and_replay(self):
        out = Path(self.tmp.name) / "demo"
        result = demo(out)
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual(result, demo(out))
        for label in ("topline", "profile", "ranking-fixed", "ranking-open"):
            run = json.loads((out / (label + "-run.json")).read_text(encoding="utf-8"))
            validate_schema(run["response"], "agent-api-response.schema.json")
            self.assertTrue(run["research"]["bundle_hash"])

    def test_reused_request_id_rejected(self):
        self.archive.bind_request("e", "same", "hash-a")
        with self.assertRaises(ArchiveConflict):
            self.archive.bind_request("e", "same", "hash-b")


if __name__ == "__main__":
    unittest.main()
