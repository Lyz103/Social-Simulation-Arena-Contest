"""Source adapters producing timestamped history and evidence for existing retrieval.

These adapters preserve source/population/unit. They never invent publication
dates from observation dates. Source manifests are producer assertions.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, build_opener

from .retrieval.providers import _NoRedirect
from .retrieval.utils import canonical_url, iso, parse_instant, utcnow
from .history import numeric
from .ranking import normalize_title


class StructuredRetriever:
    """Search imported source packages using the same snapshot filtering policy."""

    def __init__(self, paths):
        from .retrieval.providers import SnapshotRetriever
        from .retrieval.utils import digest

        packages = [
            json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in paths
        ]
        corpus = {
            "documents": [
                doc for package in packages for doc in package["corpus"]["documents"]
            ],
            "synthetic": any(
                package["corpus"].get("synthetic", False) for package in packages
            ),
            "source_packages_hash": digest(packages),
        }
        self._snapshot = SnapshotRetriever.from_corpus(corpus)

    def identity(self):
        return {
            **self._snapshot.identity(),
            "name": "structured-source-packages",
            "version": 1,
        }

    def supports(self, mode):
        return self._snapshot.supports(mode)

    def search(self, query, request, **kwargs):
        return self._snapshot.search(query, request, **kwargs)


def _manifest(manifest, raw):
    for key in ("source_name", "source_url", "captured_at", "unit", "population"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("source manifest requires " + key)
    parse_instant(manifest["captured_at"])
    canonical_url(manifest["source_url"])
    fingerprint = "sha256:" + hashlib.sha256(raw).hexdigest()
    if manifest.get("content_hash") and manifest["content_hash"] != fingerprint:
        raise ValueError("source bytes do not match manifest hash")
    return {**manifest, "content_hash": fingerprint}


def series_csv(
    path,
    manifest,
    *,
    date_column="date",
    value_column="value",
    cell_column=None,
    available_column="available_at",
):
    raw = Path(path).read_bytes()
    meta = _manifest(manifest, raw)
    rows = []
    for record in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        available = record.get(available_column) or meta.get("available_at")
        if not available:
            raise ValueError(
                "series CSV requires release availability per row or in manifest"
            )
        # Capture at a later time does not prove the original bytes existed earlier.
        row = {
            "date": record[date_column],
            "available_at": available,
            "captured_at": meta["captured_at"],
            "value": (
                {record[cell_column]: numeric(float(record[value_column]))}
                if cell_column
                else numeric(float(record[value_column]))
            ),
            "source": meta["source_name"],
            "source_url": meta["source_url"],
            "content_hash": meta["content_hash"],
            "provenance": "source_capture_asserted",
        }
        rows.append(row)
    return _package(rows, meta)


def wikipedia_snapshot(path, manifest):
    raw = Path(path).read_bytes()
    meta = _manifest(manifest, raw)
    meta["frequency"] = "daily"
    data = json.loads(raw)
    rows, titles = [], []
    spec = {"kind": "wiki_top10"}
    for item in data["items"]:
        articles = sorted(item["articles"], key=lambda a: (-a["views"], a["article"]))
        valid, views = [], {}
        for article in articles:
            try:
                name = normalize_title(article["article"], spec)
            except ValueError:
                continue
            if name not in views:
                valid.append(name)
                views[name] = numeric(article["views"])
        day = f"{item['year']}-{item['month']}-{item['day']}"
        rows.append(
            {
                "date": day,
                "available_at": meta.get("available_at", meta["captured_at"]),
                "captured_at": meta["captured_at"],
                "value": valid,
                "views": views,
                "source_url": meta["source_url"],
                "content_hash": meta["content_hash"],
                "provenance": "source_capture_asserted",
            }
        )
        titles.extend(valid[:60])
    return _package(rows, meta, list(dict.fromkeys(titles)))


def wikipedia_week(packages, *, week_start, length=10):
    """Aggregate seven captured daily top lists; never pretend absent days are zero.

    Daily API top lists are truncated. This reproduces aggregation over those
    archived lists, and explicitly records that limitation, not full pageviews.
    """
    from datetime import date, timedelta
    from .retrieval.utils import digest

    start = date.fromisoformat(week_start)
    if start.weekday() != 0 or not 2 <= length <= 50:
        raise ValueError("Wikipedia week must begin Monday; length must be 2..50")
    days = [(start + timedelta(days=i)).isoformat() for i in range(7)]
    by_day = {}
    for package in packages:
        if package["manifest"].get("frequency") != "daily":
            raise ValueError("weekly aggregation requires daily snapshots")
        for row in package["history"]:
            if row["date"] in days:
                if row["date"] in by_day and row != by_day[row["date"]]:
                    raise ValueError(
                        "multiple different captures for same day; select a revision explicitly"
                    )
                by_day[row["date"]] = row
    if set(by_day) != set(days):
        raise ValueError("weekly archive is missing one or more daily observations")
    totals = {}
    for row in by_day.values():
        for title, count in row["views"].items():
            totals[title] = totals.get(title, 0) + numeric(count)
    ranking = sorted(totals, key=lambda title: (-totals[title], title))[:length]
    if len(ranking) != length:
        raise ValueError("not enough weekly candidates")
    from .retrieval.utils import source_time_upper

    available = iso(
        max(source_time_upper(row["available_at"]) for row in by_day.values())
    )
    captured = iso(max(parse_instant(row["captured_at"]) for row in by_day.values()))
    meta = dict(packages[0]["manifest"])
    meta.update(
        frequency="weekly",
        captured_at=captured,
        available_at=available,
        content_hash=digest([by_day[d] for d in days]),
        aggregation="sum of archived daily top-list counts; unlisted daily articles contribute no observed counts",
        week_start=days[0],
        week_end=days[-1],
    )
    row = {
        "date": days[-1],
        "available_at": available,
        "captured_at": captured,
        "value": ranking,
        "views": {x: totals[x] for x in ranking},
        "source_url": meta["source_url"],
        "content_hash": meta["content_hash"],
        "provenance": "source_capture_asserted",
    }
    return _package([row], meta, ranking)


def trends_csv(
    path,
    manifest,
    *,
    items,
    date_column="date",
    available_column="available_at",
    shares=False,
):
    """One jointly normalized basket export; independently scaled queries are refused."""
    if not manifest.get("joint_basket"):
        raise ValueError("Trends comparison requires a joint_basket manifest")
    raw = Path(path).read_bytes()
    meta = _manifest(manifest, raw)
    rows = []
    for record in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        available = record.get(available_column) or meta.get("available_at")
        if not available:
            raise ValueError("Trends row requires available_at")
        values = {item: numeric(float(record[item])) for item in items}
        if any(v < 0 for v in values.values()):
            raise ValueError("Trends indices cannot be negative")
        if shares:
            total = sum(values.values())
            if total <= 0:
                raise ValueError("cannot normalize zero Trends basket")
            value = {item: 100 * v / total for item, v in values.items()}
        else:
            order = {item: i for i, item in enumerate(items)}
            value = sorted(items, key=lambda item: (-values[item], order[item]))
        rows.append(
            {
                "date": record[date_column],
                "available_at": available,
                "captured_at": meta["captured_at"],
                "value": value,
                "index": values,
                "source_url": meta["source_url"],
                "content_hash": meta["content_hash"],
            }
        )
    return _package(rows, meta)


def _package(rows, meta, candidates=None):
    from .retrieval.utils import source_time_upper

    captured = parse_instant(meta["captured_at"])
    for row in rows:
        if source_time_upper(row["available_at"]) > captured:
            raise ValueError(
                "observation availability is later than its declared capture; source manifest is inconsistent"
            )
    return {
        "manifest": meta,
        "history": rows,
        "corpus": {
            "synthetic": False,
            "documents": [
                {
                    "title": meta["source_name"],
                    "url": meta["source_url"],
                    "published_at": meta.get("published_at"),
                    "captured_at": meta["captured_at"],
                    "content": json.dumps(
                        {
                            "unit": meta["unit"],
                            "population": meta["population"],
                            "observations": rows,
                        },
                        ensure_ascii=False,
                    ),
                    "observations": rows,
                    "candidate_items": candidates or [],
                }
            ],
        },
    }


def merge_source_history(request, packages):
    """Explicitly attach compatible source packages before shared time filtering.

    A package must declare the target source, population, unit and metric. Other
    polls and indicators belong in research context, not the target history.
    """
    from dataclasses import replace
    from .retrieval.utils import digest

    rows = list(request.history)
    hashes = []
    for package in packages:
        meta = package["manifest"]
        from .ranking import is_wikipedia

        if (
            request.target_type == "ranking_list"
            and is_wikipedia(request.ranking_spec)
            and meta.get("frequency") != "weekly"
        ):
            raise ValueError(
                "weekly Wikipedia target cannot use daily rankings as weekly ground truth"
            )
        for key in ("source_name", "population", "unit", "metric"):
            expected = getattr(request, key)
            if not expected or meta.get(key) != expected:
                raise ValueError("source-history target definition mismatch: " + key)
        if not isinstance(package.get("history"), list):
            raise ValueError("source package history must be a list")
        rows.extend(package["history"])
        hashes.append(digest(package))
    metadata = dict(request.metadata)
    metadata["source_package_hashes"] = hashes
    return replace(request, history=tuple(rows), metadata=metadata)


def fetch_wikipedia_daily(
    day, *, contact_user_agent, timeout=30, clock=utcnow, transport=None
):
    """Fetch present bytes; returned capture timestamp is NOW even for old dates."""
    from datetime import date

    d = date.fromisoformat(day)
    if not contact_user_agent.strip():
        raise ValueError("Wikimedia requires a descriptive contact User-Agent")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia.org/all-access/"
        + d.strftime("%Y/%m/%d")
    )
    req = Request(url, headers={"User-Agent": contact_user_agent})
    opener = transport or build_opener(_NoRedirect()).open
    with opener(req, timeout=timeout) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Wikimedia response too large")
    json.loads(raw)
    return raw, {
        "source_name": "Wikimedia daily pageviews",
        "source_url": url,
        "captured_at": iso(clock()),
        "unit": "pageviews",
        "population": "English Wikipedia all-access readers",
        "content_hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
