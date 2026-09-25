"""Open-list candidate discovery, canonicalization and rank aggregation helpers."""

from dataclasses import dataclass
from urllib.parse import unquote, urlsplit
import math
import re


def is_wikipedia(spec):
    return (
        spec.get("kind") == "wiki_top10"
        or "wikipedia" in str(spec.get("project", "")).lower()
        or "Wikipedia" in spec.get("item_format", "")
    )


def normalize_title(value, spec):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ranking item must be a nonempty string")
    title = value.strip()
    if is_wikipedia(spec):
        title = unquote(title).replace(" ", "_")
        title = re.sub("_+", "_", title)
        title = title[0].upper() + title[1:]
    if len(title) > 256 or any(ord(c) < 32 for c in title):
        raise ValueError("invalid ranking title")
    exclusions = spec.get("exclusions", {})
    if isinstance(exclusions, str):
        exclusions = {"rule": exclusions}
    banned = set(exclusions.get("titles", []))
    prefixes = tuple(str(p).casefold() for p in exclusions.get("prefixes", []))
    if is_wikipedia(spec):
        banned.add("Main_Page")
        prefixes += tuple(
            p + ":"
            for p in (
                "special",
                "talk",
                "user",
                "user_talk",
                "wikipedia",
                "file",
                "mediawiki",
                "template",
                "help",
                "category",
                "portal",
                "draft",
                "module",
                "book",
                "media",
                "timedtext",
                "topic",
            )
        )
    if title in banned or (prefixes and title.casefold().startswith(prefixes)):
        raise ValueError("excluded ranking item")
    items = spec.get("items")
    if items and title not in items:
        raise ValueError("item is outside fixed basket")
    return title


@dataclass(frozen=True)
class CandidatePool:
    items: tuple[str, ...]
    provenance: dict[str, list[str]]
    warnings: tuple[str, ...]


class CandidateBuilder:
    VERSION = "candidates-v1"

    def __init__(self, max_candidates=60):
        if type(max_candidates) is not int or max_candidates < 2:
            raise ValueError("invalid candidate budget")
        self.max_candidates = max_candidates

    def build(self, request, documents=()):
        if request.ranking_spec.get("items"):
            return CandidatePool(
                tuple(request.candidates),
                {c: ["official_fixed_basket"] for c in request.candidates},
                (),
            )
        scores, sources, warnings = {}, {}, []

        def add(raw, score, source):
            try:
                title = normalize_title(raw, request.ranking_spec)
            except ValueError:
                warnings.append(f"excluded candidate from {source}")
                return
            scores[title] = scores.get(title, 0) + score
            sources.setdefault(title, []).append(source)

        for i, row in enumerate(request.history):
            for rank, item in enumerate(row["value"]):
                add(item, (i + 1) / (rank + 1), f"history:{row['date']}")
        for item in request.candidates:
            add(item, 1, "caller_candidate_pool")
        for doc in documents:
            ref = doc.get("evidence_id", doc.get("url", "structured"))
            for item in doc.get("candidate_items", []):
                add(item, 2, ref)
            parts = urlsplit(doc.get("url", ""))
            if (
                is_wikipedia(request.ranking_spec)
                and parts.hostname
                == request.ranking_spec.get("project", "en.wikipedia.org")
                and parts.path.startswith("/wiki/")
            ):
                add(parts.path[len("/wiki/") :], 1, ref)
        order = sorted(scores, key=lambda c: (-scores[c], c))[: self.max_candidates]
        if len(order) < request.ranking_spec["length"]:
            warnings.append(
                "candidate pool is smaller than output length; model must discover additional titles from supplied evidence"
            )
        return CandidatePool(
            tuple(order), {c: sources[c] for c in order}, tuple(warnings)
        )


def rank_scores(scores, order=None):
    if not scores or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("finite nonempty scores required")
    ties = {v: i for i, v in enumerate(order or [])}
    return sorted(scores, key=lambda c: (-scores[c], ties.get(c, len(ties)), c))
