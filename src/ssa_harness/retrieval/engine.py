"""Orchestration independent of forecasting, registration and model vendors."""

from dataclasses import asdict
import re
import time
from urllib.parse import urlsplit

from .archive import ArchiveStore
from .errors import (
    BudgetExceeded,
    EmptyCorpus,
    ProviderError,
    TemporalPolicyError,
    ValidationError,
)
from .filters import deduplicate, normalize_document, rejection_reason
from .models import EngineConfig, ResearchBundle, ResearchRequest
from .planner import TemplateQueryPlanner, normalize_queries
from .ranker import rank_documents
from .renderer import render
from .utils import digest, domain_matches, iso, parse_instant, utcnow

PIPELINE_VERSION = "ssa-harness-retrieval-v2"


class ResearchEngine:
    def __init__(
        self,
        *,
        retriever,
        store: ArchiveStore,
        planner=None,
        config=None,
        clock=utcnow,
        monotonic=time.monotonic,
    ):
        self.retriever, self.store = retriever, store
        self.planner = planner or TemplateQueryPlanner()
        self.config = config or EngineConfig()
        self.clock, self.monotonic = clock, monotonic

    def retrieve(self, request: ResearchRequest, *, entrant="default"):
        if not isinstance(entrant, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", entrant
        ):
            raise ValidationError("entrant 格式无效")
        # Freeze caller-owned mutable dictionaries before hashing and processing.
        request = ResearchRequest.from_dict(request.to_dict())
        parameters = {
            "pipeline_version": PIPELINE_VERSION,
            "config": asdict(self.config),
            "planner": self.planner.identity(),
            "retriever": self.retriever.identity(),
        }
        request_hash, pipeline_hash = digest(request.to_dict()), digest(parameters)
        existing = self.store.load_bundle(
            request.round_id, entrant, request_hash, pipeline_hash
        )
        if existing is not None:
            # Exact replay, including after deadline; never new web research.
            return existing
        with self.store.claim(
            request.round_id, entrant, self.config.max_elapsed_seconds + 60
        ):
            existing = self.store.load_bundle(
                request.round_id, entrant, request_hash, pipeline_hash
            )
            if existing is not None:
                return existing
            return self._run(request, entrant, request_hash, pipeline_hash, parameters)

    def _run(self, request, entrant, request_hash, pipeline_hash, parameters):
        asof, deadline = parse_instant(request.asof), parse_instant(request.deadline)
        now = self.clock()
        if asof > now:
            raise TemporalPolicyError("asof 不能在未来")
        if request.mode == "live" and now >= deadline:
            raise TemporalPolicyError(
                "已超过 deadline；只允许读取原有归档，不允许重新联网检索"
            )
        if not self.retriever.supports(request.mode):
            raise TemporalPolicyError("此检索器不支持请求的时间模式")
        started = self.monotonic()

        def remaining():
            seconds = self.config.max_elapsed_seconds - (self.monotonic() - started)
            if request.mode == "live":
                seconds = min(seconds, (deadline - self.clock()).total_seconds())
            if seconds <= 0:
                raise BudgetExceeded("检索总时间预算或提交截止时间已到")
            return seconds

        queries = normalize_queries(self.planner.plan(request), self.config.max_queries)
        remaining()
        documents, exclusions, retrieval_log, warnings = [], [], [], []
        failed = hits = 0
        for query in queries:
            remaining()
            cache_key = digest(
                {
                    "query": query.to_dict(),
                    "asof": request.asof,
                    "mode": request.mode,
                    "config": asdict(self.config),
                    "retriever": self.retriever.identity(),
                }
            )
            raw = self.store.cache_get(cache_key, self.config.cache_ttl_seconds)
            cached = raw is not None
            if cached:
                hits += 1
            else:
                try:
                    raw = self.retriever.search(
                        query,
                        request,
                        max_results=self.config.results_per_query,
                        lookback_days=self.config.lookback_days,
                        timeout=min(remaining(), self.config.query_timeout_seconds),
                    )
                    if not isinstance(raw, list):
                        raise ProviderError("检索器必须返回列表")
                    raw = raw[: self.config.results_per_query]
                    remaining()
                    self.store.cache_put(cache_key, raw)
                except ProviderError as exc:
                    if not self.config.allow_partial:
                        raise
                    failed += 1
                    warnings.append(
                        f"查询失败：{type(exc).__name__}；query={query.text}"
                    )
                    retrieval_log.append(
                        {
                            "query": query.to_dict(),
                            "cache_hit": False,
                            "error_type": type(exc).__name__,
                            "results": [],
                        }
                    )
                    continue
            retrieval_log.append(
                {"query": query.to_dict(), "cache_hit": cached, "results": raw}
            )
            for rank, item in enumerate(raw[: self.config.results_per_query], 1):
                doc, reason = normalize_document(item, query, rank)
                if doc is not None:
                    apply_lookback = query.topic == "news" and query.intent not in (
                        "history",
                        "measurement",
                        "crosstabs",
                    )
                    reason = rejection_reason(
                        doc,
                        request,
                        self.config.lookback_days,
                        apply_lookback=apply_lookback,
                    )
                    if (
                        reason is None
                        and parse_instant(doc["captured_at"]) > self.clock()
                    ):
                        reason = "capture_in_future"
                    if (
                        reason is None
                        and request.mode == "live"
                        and parse_instant(doc["captured_at"]) >= deadline
                    ):
                        reason = "capture_after_deadline"
                    if (
                        reason is None
                        and query.domains
                        and not domain_matches(
                            urlsplit(doc["url"]).hostname, query.domains
                        )
                    ):
                        reason = "outside_query_domains"
                if reason:
                    exclusions.append(
                        {
                            "query": query.text,
                            "url": item.get("url") if isinstance(item, dict) else None,
                            "reason": reason,
                        }
                    )
                else:
                    documents.append(doc)
        documents, duplicates = deduplicate(documents)
        exclusions.extend(duplicates)
        if not documents:
            raise EmptyCorpus("所有查询均无合格证据；不会冻结空 bundle 或自动生成预测")
        ranked = rank_documents(documents, request)
        rendered, selected, omitted = render(ranked, request, self.config)
        exclusions.extend(omitted)
        assurance = "snapshot_capture_asserted"
        if request.mode == "live":
            assurance = "publication_metadata_only"
            warnings.append(
                "实时搜索按发布时间过滤，不能证明网页在 asof 时的内容；不得将其视为严格历史回测。"
            )
        if self.retriever.identity().get("synthetic"):
            assurance = "synthetic_demo"
            warnings.append("本次使用虚构演示资料，禁止将其视为真实赛事证据。")
        remaining()
        bundle = ResearchBundle(
            schema_version="research-bundle-v1",
            round_id=request.round_id,
            entrant=entrant,
            request_hash=request_hash,
            pipeline_hash=pipeline_hash,
            request=request.to_dict(),
            params=parameters,
            created_at=iso(self.clock()),
            status="partial" if failed else "complete",
            temporal_assurance=assurance,
            queries=[q.to_dict() for q in queries],
            documents=selected,
            exclusions=exclusions,
            retrieval_log=retrieval_log,
            rendered_text=rendered,
            warnings=warnings,
            stats={
                "queries": len(queries),
                "cache_hits": hits,
                "failed_queries": failed,
                "documents": len(selected),
                "excluded": len(exclusions),
                "context_chars": len(rendered),
                "elapsed_seconds": round(self.monotonic() - started, 3),
            },
        ).seal()
        self.store.save_bundle(bundle)
        return bundle
