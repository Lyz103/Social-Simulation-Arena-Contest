"""Search adapters. Only TavilyRetriever performs network requests."""

import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .errors import MissingAPIKey, ProviderError, TemporalPolicyError, ValidationError
from .utils import digest, iso, parse_instant, utcnow


class Retriever(Protocol):
    def identity(self) -> dict: ...
    def supports(self, mode: str) -> bool: ...
    def search(
        self, query, request, *, max_results: int, lookback_days: int, timeout: float
    ) -> list[dict]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TavilyRetriever:
    """One bounded search request with limited transient retries.

    Publication metadata is only a heuristic for old cutoffs. This provider
    deliberately cannot serve snapshot mode, even with a date filter.
    """

    def __init__(
        self,
        api_key=None,
        *,
        max_attempts=3,
        transport=None,
        clock=utcnow,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValidationError("max_attempts 必须是 1..5")
        self.api_key = api_key
        self.max_attempts = max_attempts
        self.transport = transport or build_opener(_NoRedirect()).open
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep

    def identity(self):
        return {
            "name": "tavily",
            "version": 2,
            "depth": "basic",
            "max_attempts": self.max_attempts,
            "raw_content": False,
        }

    def supports(self, mode):
        return mode == "live"

    def search(self, query, request, *, max_results, lookback_days, timeout):
        if request.mode != "live":
            raise TemporalPolicyError("历史回放不能调用实时 Tavily 搜索")
        key = self.api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise MissingAPIKey("请设置 TAVILY_API_KEY；离线演示不需要密钥")
        payload = {
            "query": query.text,
            "topic": query.topic,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_published_date": True,
            "filter_by_published_date": True,
            "auto_parameters": False,
        }
        # Provider dates are coarse estimates, NOT a historical asof guarantee.
        # Include the cutoff day; the engine checks exact UTC timestamps again.
        cutoff = parse_instant(request.asof)
        payload["end_date"] = (cutoff + timedelta(days=1)).date().isoformat()
        if query.topic == "news" and query.intent not in (
            "history",
            "measurement",
            "crosstabs",
        ):
            payload["start_date"] = (
                (cutoff - timedelta(days=lookback_days)).date().isoformat()
            )
        if query.domains:
            payload["include_domains"] = list(query.domains)
        http_request = Request(
            "https://api.tavily.com/search",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        end = self.monotonic() + timeout
        for attempt in range(self.max_attempts):
            remaining = end - self.monotonic()
            if remaining <= 0:
                raise ProviderError("Tavily 查询时间预算耗尽")
            try:
                with self.transport(http_request, timeout=remaining) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ProviderError("搜索响应超过 2 MB 上限")
                data = json.loads(raw)
                if not isinstance(data, dict) or not isinstance(
                    data.get("results"), list
                ):
                    raise ProviderError("搜索响应缺少 results 列表")
                captured_at = iso(self.clock())
                items = []
                for rank, item in enumerate(data["results"][:max_results], 1):
                    if not isinstance(item, dict):
                        continue
                    items.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "content": item.get("content", ""),
                            "published_at": item.get("published_date"),
                            "modified_at": item.get("modified_at"),
                            "captured_at": captured_at,
                            "provider": "tavily",
                            "rank": rank,
                            "capture_kind": "live_search_snippet",
                            "source_metadata": {"score": item.get("score")},
                        }
                    )
                return items
            except HTTPError as exc:
                # Do not include provider response text: it can echo credentials.
                status = exc.code
                retry_after = (
                    exc.headers.get("Retry-After", "0") if exc.headers else "0"
                )
                exc.close()
                if (
                    status not in (429, 500, 502, 503, 504)
                    or attempt == self.max_attempts - 1
                ):
                    raise ProviderError(f"Tavily HTTP {status}") from None
                try:
                    delay = float(retry_after)
                    if not math.isfinite(delay):
                        delay = 0
                except ValueError:
                    delay = 0
                delay = min(5, max(0.5 * 2**attempt, delay))
            except (URLError, TimeoutError, OSError):
                if attempt == self.max_attempts - 1:
                    raise ProviderError("Tavily 网络错误或超时") from None
                delay = 0.5 * 2**attempt
            except (ValueError, UnicodeDecodeError):
                raise ProviderError("Tavily 响应不是有效 JSON") from None
            if self.monotonic() + delay >= end:
                raise ProviderError("没有足够时间重试 Tavily")
            self.sleep(delay)
        raise ProviderError("Tavily 重试失败")


class SnapshotRetriever:
    """Search a local captured corpus; never reaches the network.

    captured_at is an assertion by the archive producer, not an independent
    timestamp proof. The caller is responsible for archive provenance.
    """

    def __init__(self, path):
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict) or not isinstance(value["documents"], list):
                raise ValueError("invalid document list")
            self.documents = value["documents"]
            if any(not isinstance(d, dict) for d in self.documents):
                raise ValueError("invalid document")
            self.synthetic = bool(value.get("synthetic", False))
            self.corpus_hash = digest(value)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValidationError(
                "无法读取快照；需要包含 documents 列表的 JSON"
            ) from exc

    @classmethod
    def from_corpus(cls, value):
        """Create a snapshot from an in-memory, JSON-serializable corpus."""
        if not isinstance(value, dict) or not isinstance(value.get("documents"), list):
            raise ValidationError("snapshot corpus requires a documents list")
        if any(not isinstance(d, dict) for d in value["documents"]):
            raise ValidationError("snapshot documents must be objects")
        frozen = json.loads(json.dumps(value, allow_nan=False))
        instance = cls.__new__(cls)
        instance.documents = frozen["documents"]
        instance.synthetic = bool(frozen.get("synthetic", False))
        instance.corpus_hash = digest(frozen)
        return instance

    def identity(self):
        return {
            "name": "local-snapshot",
            "version": 1,
            "corpus_hash": self.corpus_hash,
            "synthetic": self.synthetic,
        }

    def supports(self, mode):
        return mode in ("snapshot", "live")

    def search(self, query, request, *, max_results, lookback_days, timeout):
        from .filters import normalize_document, rejection_reason
        from .ranker import relevance

        candidates = []
        # Filter before top-k so a future document cannot hide valid old ones.
        for index, raw in enumerate(self.documents):
            normalized, _ = normalize_document(raw, query, index + 1)
            apply_lookback = query.topic == "news" and query.intent not in (
                "history",
                "measurement",
                "crosstabs",
            )
            if normalized is None or rejection_reason(
                normalized, request, lookback_days, apply_lookback=apply_lookback
            ):
                continue
            from urllib.parse import urlsplit
            from .utils import domain_matches

            if query.domains and not domain_matches(
                urlsplit(normalized["url"]).hostname, query.domains
            ):
                continue
            score = relevance(
                query.text, normalized["title"] + " " + normalized["content"]
            )
            if score > 0:
                candidates.append((score, index, raw))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [
            dict(raw, provider="local-snapshot", capture_kind="local_snapshot")
            for _, _, raw in candidates[:max_results]
        ]
