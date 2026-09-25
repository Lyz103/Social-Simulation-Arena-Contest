"""Normalize content and explicitly report exclusions."""

from datetime import timedelta

from .errors import ValidationError
from .utils import canonical_url, clean_text, digest, parse_instant, source_time_upper


def normalize_document(raw, query, rank):
    if not isinstance(raw, dict):
        return None, "invalid_document"
    try:
        url = canonical_url(raw.get("url"))
    except ValidationError:
        return None, "invalid_url"
    content = clean_text(raw.get("content") or raw.get("snippet"))
    if not content:
        return None, "empty_content"
    content = content[:100000]
    result = {
        "title": clean_text(raw.get("title"))[:500] or "Untitled",
        "url": url,
        "original_url": raw["url"],
        "content": content,
        "published_at": raw.get("published_at") or raw.get("published_date"),
        "modified_at": raw.get("modified_at"),
        "captured_at": raw.get("captured_at"),
        "capture_kind": raw.get("capture_kind", "unspecified"),
        "provider": raw.get("provider", "custom"),
        "raw_hash": digest(raw),
        "content_hash": digest(content.casefold()),
        "matches": [{"query": query.text, "intent": query.intent, "rank": rank}],
    }
    for key in ("candidate_items", "observations", "stance"):
        if key in raw:
            result[key] = raw[key]
    return result, None


def rejection_reason(document, request, lookback_days, *, apply_lookback=True):
    cutoff = parse_instant(request.asof)
    try:
        captured = parse_instant(document["captured_at"])
    except ValidationError:
        return "missing_or_invalid_capture_time"
    if request.mode == "snapshot" and captured > cutoff:
        return "capture_after_cutoff"
    published = document["published_at"]
    if published:
        try:
            upper = source_time_upper(published)
        except ValidationError:
            return "invalid_publication_time"
        if upper > cutoff:
            return "publication_after_cutoff_or_ambiguous_day"
        if apply_lookback and upper < cutoff - timedelta(days=lookback_days):
            return "outside_lookback_window"
    elif request.mode == "live":
        return "missing_publication_time"
    # Snapshot bytes captured before asof remain usable without publication date.
    if document["modified_at"]:
        try:
            if source_time_upper(document["modified_at"]) > cutoff:
                return "modified_after_cutoff"
        except ValidationError:
            return "invalid_modification_time"
    return None


def deduplicate(documents):
    by_url, by_content, kept, exclusions = {}, {}, [], []
    for doc in documents:
        existing = by_url.get(doc["url"]) or by_content.get(doc["content_hash"])
        if existing is not None:
            existing["matches"].extend(doc["matches"])
            exclusions.append(
                {"url": doc["url"], "reason": "duplicate", "raw_hash": doc["raw_hash"]}
            )
        else:
            kept.append(doc)
            by_url[doc["url"]] = doc
            by_content[doc["content_hash"]] = doc
    return kept, exclusions
