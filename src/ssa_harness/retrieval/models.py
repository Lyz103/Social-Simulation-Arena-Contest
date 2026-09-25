"""Public contracts independent of any competition repository."""

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from .errors import ValidationError, ArchiveCorruption
from .utils import canonical_json, digest, iso, parse_instant

TARGETS = {"continuous_normal", "profile_energy", "ranking_list"}


def _strings(values, field_name, limit):
    if not isinstance(values, (list, tuple)) or len(values) > limit:
        raise ValidationError(f"{field_name} 必须是至多 {limit} 项的字符串列表")
    if any(not isinstance(x, str) or not x.strip() or len(x) > 1000 for x in values):
        raise ValidationError(f"{field_name} 包含空值或无效字符串")
    return tuple(values)


@dataclass(frozen=True)
class ResearchRequest:
    round_id: str
    question: str
    target_type: str
    unit: str
    asof: str
    deadline: str
    mode: str = "live"
    domain: str = "general"
    source_name: str = ""
    population: str = ""
    metric: str = ""
    search_terms: str = ""
    source_domains: tuple[str, ...] = ()
    cells: tuple[str, ...] = ()
    ranking_spec: dict[str, Any] = field(default_factory=dict)
    history: Any = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    queries: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.round_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.round_id
        ):
            raise ValidationError("round_id 格式无效")
        for name in ("question", "unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 12000:
                raise ValidationError(f"{name} 不能为空且必须是合理长度的字符串")
        if self.target_type not in TARGETS:
            raise ValidationError("不支持的 target_type")
        if self.mode not in ("live", "snapshot"):
            raise ValidationError("mode 必须是 live 或 snapshot")
        for name in ("domain", "source_name", "population", "metric", "search_terms"):
            if (
                not isinstance(getattr(self, name), str)
                or len(getattr(self, name)) > 1000
            ):
                raise ValidationError(f"{name} 格式无效")
        asof, deadline = parse_instant(self.asof), parse_instant(self.deadline)
        if asof >= deadline:
            raise ValidationError("asof 必须早于 deadline")
        object.__setattr__(self, "asof", iso(asof))
        object.__setattr__(self, "deadline", iso(deadline))
        for name, limit in (("cells", 100), ("queries", 50), ("source_domains", 20)):
            object.__setattr__(self, name, _strings(getattr(self, name), name, limit))
        if len(set(self.cells)) != len(self.cells):
            raise ValidationError("cells 不允许重复")
        domains = tuple(x.lower() for x in self.source_domains)
        if any(
            not re.fullmatch(r"[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?", x) or ".." in x
            for x in domains
        ):
            raise ValidationError("source_domains 只能填写域名，不能包含协议或路径")
        object.__setattr__(self, "source_domains", domains)
        if self.target_type == "profile_energy" and not self.cells:
            raise ValidationError("Profile 题必须显式提供 cells")
        if not isinstance(self.ranking_spec, dict) or not isinstance(
            self.context, dict
        ):
            raise ValidationError("ranking_spec/context 必须是对象")
        if self.target_type == "ranking_list":
            n = self.ranking_spec.get("length")
            if type(n) is not int or not 2 <= n <= 50:
                raise ValidationError("Ranking 题必须提供 2..50 的 ranking_spec.length")
        try:
            canonical_json(asdict(self))
        except (TypeError, ValueError) as exc:
            raise ValidationError("请求必须可以序列化为有限数值的 JSON") from exc

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValidationError("request 必须是 JSON 对象")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValidationError("request 缺少字段或包含未知字段") from exc

    @classmethod
    def from_envelope(cls, envelope, *, asof, mode="live", **metadata):
        """Official v2 envelope adapter; all three history shapes are preserved."""
        if "schema_version" in envelope:
            from ..contracts import ForecastRequest

            adapted = ForecastRequest.from_envelope(envelope, asof=asof)
            value = adapted.to_research_request(mode=mode).to_dict()
            value["context"] = envelope["round"]["context"]
            value.update(metadata)
            return cls.from_dict(value)
        # Legacy bare-round input remains available to existing retrieval users.
        r = envelope.get("round", envelope)
        context = r.get("context", {})
        return cls(
            round_id=r["round_id"],
            question=r["question"],
            target_type=r["target_type"],
            unit=r["unit"],
            asof=asof,
            deadline=r.get("deadline") or r["lock_at"],
            mode=mode,
            cells=tuple(r.get("cells", ())),
            ranking_spec=r.get("ranking", {}),
            context=context,
            history=context.get("history", []),
            **metadata,
        )


@dataclass(frozen=True)
class SearchQuery:
    text: str
    intent: str = "background"
    topic: str = "news"
    domains: tuple[str, ...] = ()

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))


@dataclass(frozen=True)
class EngineConfig:
    max_queries: int = 4
    results_per_query: int = 3
    max_documents: int = 12
    snippet_chars: int = 700
    max_context_chars: int = 14000
    lookback_days: int = 14
    cache_ttl_seconds: int = 900
    max_elapsed_seconds: int = 150
    query_timeout_seconds: int = 30
    allow_partial: bool = False

    def __post_init__(self):
        limits = {
            "max_queries": (1, 20),
            "results_per_query": (1, 20),
            "max_documents": (1, 100),
            "snippet_chars": (50, 10000),
            "max_context_chars": (1500, 200000),
            "lookback_days": (1, 3650),
            "cache_ttl_seconds": (1, 86400),
            "max_elapsed_seconds": (1, 3600),
            "query_timeout_seconds": (1, 120),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValidationError(f"{name} 必须在 {low}..{high} 之间")
        if type(self.allow_partial) is not bool:
            raise ValidationError("allow_partial 必须是布尔值")


@dataclass
class ResearchBundle:
    """Serializable evidence, exclusions and exact model-visible text."""

    schema_version: str
    round_id: str
    entrant: str
    request_hash: str
    pipeline_hash: str
    request: dict
    params: dict
    created_at: str
    status: str
    temporal_assurance: str
    queries: list[dict]
    documents: list[dict]
    exclusions: list[dict]
    retrieval_log: list[dict]
    rendered_text: str
    warnings: list[str]
    stats: dict
    bundle_hash: str = ""

    def seal(self):
        self.bundle_hash = digest(self._payload())
        return self

    def _payload(self):
        value = asdict(self)
        value.pop("bundle_hash")
        return value

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))

    def verify(self):
        if (
            self.schema_version != "research-bundle-v1"
            or digest(self._payload()) != self.bundle_hash
        ):
            raise ArchiveCorruption("Research Bundle 版本或 SHA-256 校验失败")

    @classmethod
    def from_dict(cls, value):
        try:
            bundle = cls(**value)
            bundle.verify()
            return bundle
        except (TypeError, ValueError) as exc:
            raise ArchiveCorruption("Research Bundle 结构损坏") from exc
