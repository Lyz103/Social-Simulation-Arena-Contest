"""Small deterministic ranking heuristic, not a truth probability."""

import math
import re
from urllib.parse import urlsplit

from .utils import domain_matches, parse_instant, source_time_upper

AUTHORITY = {
    "yougov.com": 1.0,
    "civiqs.com": 1.0,
    "sca.isr.umich.edu": 1.0,
    "newyorkfed.org": 1.0,
    "aaii.com": 1.0,
    "wikimedia.org": 1.0,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
}


def tokens(value):
    english = re.findall(r"[a-z0-9]+", value.casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]", value)
    return set(english + chinese)


def relevance(query, text):
    terms = tokens(query)
    return len(terms & tokens(text)) / max(len(terms), 1)


def rank_documents(documents, request):
    cutoff = parse_instant(request.asof)
    for doc in documents:
        host = urlsplit(doc["url"]).hostname
        authority = max(
            [
                weight
                for domain, weight in AUTHORITY.items()
                if domain_matches(host, (domain,))
            ]
            or [0.4]
        )
        if domain_matches(host, request.source_domains):
            authority = 1.0
        timestamp = (
            source_time_upper(doc["published_at"])
            if doc["published_at"]
            else parse_instant(doc["captured_at"])
        )
        age = max(0, (cutoff - timestamp).total_seconds() / 86400)
        rel = max(
            relevance(match["query"], doc["title"] + " " + doc["content"])
            for match in doc["matches"]
        )
        quality = min(len(doc["content"]) / 500, 1)
        doc["scores"] = {
            "authority": authority,
            "freshness": round(math.exp(-age / 7), 6),
            "relevance": round(rel, 6),
            "content_length": quality,
        }
        doc["score"] = round(
            0.3 * authority + 0.25 * math.exp(-age / 7) + 0.35 * rel + 0.1 * quality, 6
        )
    return sorted(documents, key=lambda d: (-d["score"], d["url"], d["content_hash"]))
