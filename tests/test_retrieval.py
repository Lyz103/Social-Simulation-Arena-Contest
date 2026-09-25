import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from urllib.error import HTTPError

from ssa_harness.retrieval import (
    ArchiveStore,
    EngineConfig,
    JSONQueryPlanner,
    ResearchBundle,
    ResearchEngine,
    ResearchRequest,
    SearchQuery,
    SnapshotRetriever,
    TavilyRetriever,
    TemplateQueryPlanner,
)
from ssa_harness.retrieval.errors import (
    ArchiveConflict,
    ArchiveCorruption,
    BudgetExceeded,
    EmptyCorpus,
    ProviderError,
    TemporalPolicyError,
    ValidationError,
    WorkInProgress,
)
from ssa_harness.retrieval.filters import normalize_document, rejection_reason
from ssa_harness.retrieval.planner import normalize_queries
from ssa_harness.retrieval.utils import canonical_url, clean_text, iso, parse_instant

EXAMPLES = (
    Path(__import__("ssa_harness.retrieval", fromlist=["__file__"]).__file__).parent
    / "examples"
)
NOW = parse_instant("2026-09-20T13:00:00Z")


def request(**changes):
    value = dict(
        round_id="r1",
        question="Example Poll approval",
        target_type="continuous_normal",
        unit="%",
        asof="2026-09-20T12:00:00Z",
        deadline="2026-09-21T12:00:00Z",
        queries=("approval",),
        mode="live",
    )
    value.update(changes)
    return ResearchRequest(**value)


def document(**changes):
    value = dict(
        title="Example approval",
        url="https://poll.example.org/a?id=1",
        content="Approval research for the Example Poll population.",
        published_at="2026-09-19T10:00:00Z",
        captured_at="2026-09-20T11:00:00Z",
    )
    value.update(changes)
    return value


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results if results is not None else [document()]
        self.calls = 0

    def identity(self):
        return {"name": "fake", "version": 1}

    def supports(self, mode):
        return True

    def search(self, query, request, **kwargs):
        self.calls += 1
        if query.text == "fail":
            raise ProviderError("simulated failure")
        return self.results


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "archive.sqlite3"
        self.store = ArchiveStore(self.db)
        self.provider = FakeRetriever()
        self.engine = ResearchEngine(
            retriever=self.provider, store=self.store, clock=lambda: NOW
        )

    def test_three_task_shapes_offline(self):
        for kind in ("topline", "profile", "ranking"):
            req = ResearchRequest.from_dict(
                json.loads((EXAMPLES / f"{kind}.json").read_text(encoding="utf-8"))
            )
            engine = ResearchEngine(
                retriever=SnapshotRetriever(EXAMPLES / "corpus.json"),
                store=self.store,
                clock=lambda: NOW,
            )
            bundle = engine.retrieve(req, entrant="demo")
            bundle.verify()
            self.assertGreater(len(bundle.documents), 0)
            self.assertEqual(bundle.temporal_assurance, "synthetic_demo")
            self.assertNotIn("future; never", bundle.rendered_text)
            self.assertLessEqual(len(bundle.rendered_text), 14000)

    def test_repeat_after_deadline_returns_exact_bundle_without_network(self):
        first = self.engine.retrieve(request())
        self.engine.clock = lambda: NOW + timedelta(days=10)
        second = self.engine.retrieve(request())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(self.provider.calls, 1)

    def test_changed_request_cannot_overwrite(self):
        self.engine.retrieve(request())
        with self.assertRaises(ArchiveConflict):
            self.engine.retrieve(request(question="different question"))

    def test_changed_config_cannot_silently_replay(self):
        self.engine.retrieve(request())
        self.engine.config = EngineConfig(snippet_chars=300)
        with self.assertRaises(ArchiveConflict):
            self.engine.retrieve(request())

    def test_same_query_shared_but_different_cutoff_not_shared(self):
        first = self.engine.retrieve(request(), entrant="a")
        second = self.engine.retrieve(request(), entrant="b")
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(second.stats["cache_hits"], 1)
        self.engine.retrieve(request(round_id="r2", asof="2026-09-20T12:30:00Z"))
        self.assertEqual(self.provider.calls, 2)

    def test_parameters_in_cache_identity(self):
        self.engine.retrieve(request(), entrant="a")
        self.engine.config = EngineConfig(results_per_query=2)
        self.engine.retrieve(request(), entrant="b")
        self.assertEqual(self.provider.calls, 2)

    def test_future_and_missing_dates_rejected_with_reasons(self):
        self.provider.results = [
            document(),
            document(url="https://e.org/f", published_at="2026-09-22T00:00:00Z"),
            document(url="https://e.org/u", published_at=None),
        ]
        bundle = self.engine.retrieve(request())
        self.assertEqual(len(bundle.documents), 1)
        self.assertEqual(
            {d["reason"] for d in bundle.exclusions},
            {"publication_after_cutoff_or_ambiguous_day", "missing_publication_time"},
        )

    def test_future_capture_rejected(self):
        self.provider.results = [document(captured_at="2026-09-22T00:00:00Z")]
        with self.assertRaises(EmptyCorpus):
            self.engine.retrieve(request())

    def test_query_budget_enforced_even_for_custom_planner(self):
        class Planner:
            def identity(self):
                return {"name": "many"}

            def plan(self, req):
                return ["same", " SAME ", "q2", "q3", "q4", "q5"]

        self.engine.planner = Planner()
        bundle = self.engine.retrieve(request())
        self.assertEqual(self.provider.calls, 4)
        self.assertEqual(len(bundle.queries), 4)

    def test_provider_failure_does_not_freeze_bundle(self):
        req = request(queries=("approval", "fail"))
        with self.assertRaises(ProviderError):
            self.engine.retrieve(req)
        with closing(sqlite3.connect(self.db)) as db, db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM bundles").fetchone()[0], 0
            )
        self.engine.config = EngineConfig(allow_partial=True)
        bundle = self.engine.retrieve(req)
        self.assertEqual(bundle.status, "partial")
        self.assertEqual(bundle.stats["failed_queries"], 1)

    def test_empty_is_not_success_even_in_partial_mode(self):
        self.engine.config = EngineConfig(allow_partial=True)
        with self.assertRaises(EmptyCorpus):
            self.engine.retrieve(request(queries=("fail",)))

    def test_expired_live_request_refused_before_network(self):
        self.engine.clock = lambda: NOW + timedelta(days=2)
        with self.assertRaises(TemporalPolicyError):
            self.engine.retrieve(request())
        self.assertEqual(self.provider.calls, 0)

    def test_future_asof_refused(self):
        self.engine.clock = lambda: NOW - timedelta(days=1)
        with self.assertRaises(TemporalPolicyError):
            self.engine.retrieve(request())

    def test_time_budget_checked_after_provider(self):
        ticks = [0]

        class Slow(FakeRetriever):
            def search(self, *args, **kwargs):
                ticks[0] += 200
                return [document()]

        engine = ResearchEngine(
            retriever=Slow(),
            store=self.store,
            clock=lambda: NOW,
            monotonic=lambda: ticks[0],
        )
        with self.assertRaises(BudgetExceeded):
            engine.retrieve(request())

    def test_archive_tampering_is_detected(self):
        self.engine.retrieve(request())
        with closing(sqlite3.connect(self.db)) as db, db:
            raw = json.loads(db.execute("SELECT payload FROM bundles").fetchone()[0])
            raw["rendered_text"] = "tampered"
            db.execute("UPDATE bundles SET payload=?", (json.dumps(raw),))
        with self.assertRaises(ArchiveCorruption):
            self.engine.retrieve(request())

    def test_cache_tampering_is_not_silently_used(self):
        self.engine.retrieve(request(), entrant="first")
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("UPDATE query_cache SET payload='[]'")
        with self.assertRaises(ArchiveCorruption):
            self.engine.retrieve(request(), entrant="second")

    def test_two_stores_cannot_claim_same_round(self):
        other = ArchiveStore(self.db)
        with self.store.claim("r1", "a", 60):
            with self.assertRaises(WorkInProgress):
                with other.claim("r1", "a", 60):
                    self.fail("duplicate work")
        with other.claim("r1", "a", 60):
            pass

    def test_dedup_preserves_query_provenance(self):
        bundle = self.engine.retrieve(request(queries=("approval", "poll")))
        self.assertEqual(len(bundle.documents), 1)
        self.assertEqual(len(bundle.documents[0]["matches"]), 2)

    def test_context_budget_and_evidence_references(self):
        self.provider.results = [
            document(url=f"https://e.org/{i}", content=f"Item {i} " + "word " * 300)
            for i in range(3)
        ]
        self.engine.config = EngineConfig(max_context_chars=1500)
        bundle = self.engine.retrieve(request())
        self.assertLessEqual(len(bundle.rendered_text), 1500)
        self.assertIn('"evidence_id": "E1"', bundle.rendered_text)
        self.assertTrue(any(x["reason"] == "context_budget" for x in bundle.exclusions))

    def test_history_intent_can_use_older_pre_cutoff_records(self):
        old = document(published_at="2025-01-10T00:00:00Z")
        provider = FakeRetriever([old])
        engine = ResearchEngine(retriever=provider, store=self.store, clock=lambda: NOW)
        req = request(queries=("approval history",))

        class HistoryPlanner:
            def identity(self):
                return {"name": "history"}

            def plan(self, request):
                return [
                    SearchQuery("approval history", intent="history", topic="general")
                ]

        engine.planner = HistoryPlanner()
        bundle = engine.retrieve(req)
        self.assertEqual(len(bundle.documents), 1)

    def test_recent_news_still_obeys_lookback(self):
        old = document(published_at="2025-01-10T00:00:00Z")
        provider = FakeRetriever([old])
        engine = ResearchEngine(retriever=provider, store=self.store, clock=lambda: NOW)

        class NewsPlanner:
            def identity(self):
                return {"name": "news"}

            def plan(self, request):
                return [SearchQuery("approval news", intent="drivers", topic="news")]

        engine.planner = NewsPlanner()
        with self.assertRaises(EmptyCorpus):
            engine.retrieve(request(queries=("approval news",)))


class UtilityTests(unittest.TestCase):
    def test_url_keeps_meaningful_query_and_drops_trackers(self):
        self.assertEqual(
            canonical_url(
                "https://EXAMPLE.org/item?id=1&date=2026-09-01&utm_source=x#part"
            ),
            "https://example.org/item?id=1&date=2026-09-01",
        )
        self.assertNotEqual(
            canonical_url("https://e.org/?id=1"), canonical_url("https://e.org/?id=2")
        )
        with self.assertRaises(ValidationError):
            canonical_url("file:///tmp/a")

    def test_html_and_control_chars_removed(self):
        self.assertEqual(
            clean_text("<p>Hello</p><script>bad()</script><p>world\u200b</p>"),
            "Hello world",
        )

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError):
            request(asof="2026-09-20T12:00:00")
        with self.assertRaises(ValidationError):
            request(target_type="profile_energy")

    def test_old_publication_does_not_prove_old_snapshot(self):
        req = request(mode="snapshot")
        doc, _ = normalize_document(
            document(captured_at="2026-09-20T13:00:00Z"), SearchQuery("q"), 1
        )
        self.assertEqual(rejection_reason(doc, req, 14), "capture_after_cutoff")

    def test_snapshot_may_use_undated_bytes_captured_before_cutoff(self):
        doc, _ = normalize_document(document(published_at=None), SearchQuery("q"), 1)
        self.assertIsNone(rejection_reason(doc, request(mode="snapshot"), 14))
        self.assertEqual(
            rejection_reason(doc, request(), 14), "missing_publication_time"
        )

    def test_date_only_cutoff_day_is_ambiguous(self):
        doc, _ = normalize_document(
            document(published_at="2026-09-20"), SearchQuery("q"), 1
        )
        self.assertEqual(
            rejection_reason(doc, request(), 14),
            "publication_after_cutoff_or_ambiguous_day",
        )

    def test_future_event_announced_before_cutoff_is_usable(self):
        doc, _ = normalize_document(
            document(content="Movie release scheduled for October 1."),
            SearchQuery("q"),
            1,
        )
        self.assertIsNone(rejection_reason(doc, request(), 14))

    def test_json_planner_format_and_budget(self):
        planner = JSONQueryPlanner(
            lambda p: json.dumps({"queries": ["a", " A ", 3, "b", "c", "d", "e"]}),
            version="test",
        )
        self.assertEqual(
            [q.text for q in planner.plan(request())], ["a", "b", "c", "d"]
        )
        with self.assertRaises(ValidationError):
            JSONQueryPlanner(
                lambda p: "explanation instead of JSON", version="test"
            ).plan(request())

    def test_attention_profile_does_not_search_demographics(self):
        req = request(
            target_type="profile_energy",
            cells=("tesla", "iphone"),
            domain="attention",
            queries=(),
        )
        self.assertNotIn(
            "demographic", " ".join(q.text for q in TemplateQueryPlanner().plan(req))
        )


class TavilyTests(unittest.TestCase):
    def test_request_shape_and_normalization_without_network(self):
        seen = []

        def transport(req, timeout):
            seen.append((req, timeout))
            return io.BytesIO(
                json.dumps(
                    {
                        "results": [
                            dict(document(), published_date="2026-09-19T10:00:00Z")
                        ]
                    }
                ).encode()
            )

        provider = TavilyRetriever(
            "test-secret", transport=transport, clock=lambda: NOW
        )
        results = provider.search(
            SearchQuery("approval", topic="news"),
            request(),
            max_results=3,
            lookback_days=14,
            timeout=10,
        )
        payload = json.loads(seen[0][0].data)
        self.assertNotIn("api_key", payload)
        self.assertFalse(payload["include_answer"])
        self.assertTrue(payload["include_published_date"])
        self.assertTrue(payload["filter_by_published_date"])
        self.assertFalse(payload["auto_parameters"])
        self.assertEqual(payload["start_date"], "2026-09-06")
        self.assertEqual(payload["end_date"], "2026-09-21")
        self.assertNotIn("days", payload)
        self.assertEqual(results[0]["captured_at"], iso(NOW))
        self.assertEqual(results[0]["provider"], "tavily")

    def test_history_search_requests_dates_without_news_lookback(self):
        seen = []

        def transport(req, timeout):
            seen.append(json.loads(req.data))
            return io.BytesIO(b'{"results": []}')

        provider = TavilyRetriever(
            "test-secret", transport=transport, clock=lambda: NOW
        )
        for topic in ("general", "news"):
            provider.search(
                SearchQuery("approval history", intent="history", topic=topic),
                request(),
                max_results=3,
                lookback_days=14,
                timeout=10,
            )
        for payload in seen:
            self.assertTrue(payload["include_published_date"])
            self.assertNotIn("start_date", payload)
            self.assertNotIn("days", payload)
            self.assertEqual(payload["end_date"], "2026-09-21")

    def test_snapshot_refused(self):
        provider = TavilyRetriever("test")
        with self.assertRaises(TemporalPolicyError):
            provider.search(
                SearchQuery("q"),
                request(mode="snapshot"),
                max_results=3,
                lookback_days=14,
                timeout=5,
            )

    def test_transient_retry_and_no_secret_in_error(self):
        calls, delays = [], []

        def transport(req, timeout):
            calls.append(req)
            if len(calls) == 1:
                raise HTTPError(
                    req.full_url, 429, "test-secret", {"Retry-After": "0"}, None
                )
            return io.BytesIO(b'{"results": []}')

        provider = TavilyRetriever(
            "test-secret", transport=transport, sleep=delays.append
        )
        self.assertEqual(
            provider.search(
                SearchQuery("q"), request(), max_results=3, lookback_days=14, timeout=5
            ),
            [],
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(delays), 1)

        def denied(req, timeout):
            raise HTTPError(req.full_url, 401, "test-secret", {}, None)

        provider.transport = denied
        with self.assertRaisesRegex(ProviderError, "Tavily HTTP 401") as error:
            provider.search(
                SearchQuery("q"), request(), max_results=3, lookback_days=14, timeout=5
            )
        self.assertNotIn("test-secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
