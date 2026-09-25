"""First-party adapters for public polling dashboards.

The adapters deliberately return the same document shape as the retrieval
providers and can also emit a source package for a strict, reproducible run.
They do not pretend that an observation date is a publication timestamp:
live captures are stamped when the bytes are downloaded.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, build_opener
from xml.etree import ElementTree as ET

from .retrieval.errors import ProviderError, TemporalPolicyError, ValidationError
from .retrieval.utils import iso, parse_instant, utcnow


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _capture_document(
    *,
    title: str,
    url: str,
    raw: bytes,
    captured_at: str,
    content: str,
    observations=None,
    candidate_items=None,
):
    return {
        "title": title,
        "url": url,
        "content": content,
        "published_at": None,
        "modified_at": None,
        "captured_at": captured_at,
        "provider": "first-party-source",
        "capture_kind": "source_snapshot",
        "content_hash": _sha256(raw),
        "observations": observations or [],
        "candidate_items": candidate_items or [],
    }


def _bounded_get(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    transport=None,
    max_bytes: int = 12_000_000,
) -> bytes:
    req = Request(url, headers={"User-Agent": user_agent})
    # Both public sources currently use a relative redirect for their default
    # dashboard/download URL. urllib's standard opener follows it while still
    # keeping the request bounded; callers that need stricter policy can inject
    # a transport in tests.
    opener = transport or build_opener().open
    try:
        with opener(req, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
    except Exception as exc:  # urllib errors vary by platform/provider
        status = getattr(exc, "code", None)
        suffix = f" HTTP {status}" if status else ""
        raise ProviderError(
            f"source request failed{suffix}: {type(exc).__name__}"
        ) from None
    if len(raw) > max_bytes:
        raise ProviderError("source response exceeds the bounded size limit")
    return raw


def _json_object_after_marker(text: str, marker: str):
    """Parse a brace-balanced JSON object after a Remix route marker."""
    start = text.find(marker)
    if start < 0:
        return None
    begin = text.find("{", start + len(marker))
    if begin < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(begin, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[begin : index + 1])
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        "Civiqs embedded payload is invalid JSON"
                    ) from exc
    return None


def parse_civiqs_html(raw: bytes | str, *, url: str = "") -> dict:
    """Extract the Civiqs tracker payload from a captured HTML page."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    marker = '"routes/_app.results_.$question":'
    payload = _json_object_after_marker(text, marker)
    if payload is None:
        index = _json_object_after_marker(text, '"routes/_app._index":')
        if index is not None:
            raise ValidationError(
                f"Civiqs returned its index page for {url or 'request'}"
            )
        raise ValidationError(
            "Civiqs tracker payload was not found; page format may have changed"
        )
    topline = payload.get("topline") or {}
    series = topline.get("line_chart_data")
    if not isinstance(series, list) or not series:
        raise ValidationError("Civiqs payload has no line_chart_data")
    return payload


def _civiqs_points(payload: dict):
    choices = {}
    for series in payload["topline"]["line_chart_data"]:
        key = series.get("key")
        if not isinstance(key, str) or not key:
            continue
        points = []
        for value in series.get("values") or []:
            try:
                stamp = datetime.fromtimestamp(
                    float(value["date"]) / 1000.0, timezone.utc
                )
                number = round(float(value["value"]) * 100.0, 4)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            points.append({"date": stamp.date().isoformat(), "value": number})
        if points:
            choices[key] = points
    if not choices:
        raise ValidationError("Civiqs tracker contains no usable dated values")
    return choices


def _as_side(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(item) for item in value]


def civiqs_package(
    raw: bytes, *, tracker: str, url: str, captured_at: str, filters=None
):
    payload = parse_civiqs_html(raw, url=url)
    choices = _civiqs_points(payload)
    job = payload.get("job_description") or {}
    display_net = job.get("display_net") or {}
    minuend = _as_side(display_net.get("minuend"))
    subtrahend = _as_side(display_net.get("subtrahend"))
    if not minuend and "Approve" in choices and "Disapprove" in choices:
        minuend, subtrahend = ["Approve"], ["Disapprove"]
    observations = []
    dates = sorted({point["date"] for points in choices.values() for point in points})
    for day in dates:
        values = {
            name: next((p["value"] for p in points if p["date"] == day), None)
            for name, points in choices.items()
        }
        # Keep the full marginals. A net is computed only if both sides are complete.
        value = None
        if (
            minuend
            and subtrahend
            and all(values.get(name) is not None for name in minuend + subtrahend)
        ):
            value = round(
                sum(values[name] for name in minuend)
                - sum(values[name] for name in subtrahend),
                4,
            )
        observations.append(
            {
                "date": day,
                "available_at": captured_at,
                "captured_at": captured_at,
                "value": value if value is not None else values,
                "shares": values,
                "source": "Civiqs",
                "source_url": url,
                "content_hash": _sha256(raw),
                "provenance": "civiqs_page_capture",
            }
        )
    title = job.get("display_text") or tracker
    meta = {
        "source_name": f"Civiqs: {title}",
        "source_url": url,
        "captured_at": captured_at,
        "available_at": captured_at,
        "unit": "percentage points",
        "population": job.get("population_model") or "Civiqs modeled registered voters",
        "metric": "Civiqs displayed series/net",
        "tracker": tracker,
        "filters": dict(filters or {}),
        "run_id": payload.get("run_id"),
        "end_date": payload.get("end_date"),
        "job_finish_time": payload.get("job_finish_time"),
        "display_net": {"minuend": minuend, "subtrahend": subtrahend},
        "content_hash": _sha256(raw),
    }
    content = json.dumps(
        {
            "tracker": tracker,
            "filters": filters or {},
            "choices": choices,
            "metadata": meta,
        },
        ensure_ascii=False,
    )
    document = _capture_document(
        title=title,
        url=url,
        raw=raw,
        captured_at=captured_at,
        content=content,
        observations=observations,
        candidate_items=[],
    )
    document["source_metadata"] = meta
    return {
        "manifest": meta,
        "history": observations,
        "corpus": {"synthetic": False, "documents": [document]},
    }


class CiviqsRetriever:
    """Live Civiqs dashboard retriever; no API key is used."""

    def __init__(
        self,
        tracker="approve_president_trump_2025",
        *,
        filters=None,
        api_base="https://civiqs.com/results",
        user_agent="MIT-SSA-research/0.1 (contact project maintainer)",
        timeout=60,
        transport=None,
        clock=utcnow,
    ):
        self.tracker = tracker
        self.filters = dict(filters or {})
        self.api_base = api_base.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.transport = transport
        self.clock = clock

    def identity(self):
        return {
            "name": "civiqs",
            "version": 1,
            "tracker": self.tracker,
            "filters": self.filters,
            "api_key": False,
        }

    def supports(self, mode):
        return mode == "live"

    def search(self, query, request, *, max_results, lookback_days, timeout):
        if request.mode != "live":
            raise TemporalPolicyError(
                "Civiqs live dashboard cannot prove a historical snapshot"
            )
        url = f"{self.api_base}/{self.tracker}"
        if self.filters:
            url += "?" + urlencode(sorted(self.filters.items()))
        raw = _bounded_get(
            url,
            user_agent=self.user_agent,
            timeout=min(timeout, self.timeout),
            transport=self.transport,
        )
        captured_at = iso(self.clock())
        package = civiqs_package(
            raw,
            tracker=self.tracker,
            url=url,
            captured_at=captured_at,
            filters=self.filters,
        )
        return [package["corpus"]["documents"][0]]


_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_WORKSHEET_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
)
_YOU_GOV_URL = (
    "https://api-test.yougov.com/public-data/v5/us/trackers/{tracker}/download/"
)
_YOU_GOV_TOPLINE = "US Registered Voters"
_YOU_GOV_CELLS = [
    "Democrat",
    "Independent",
    "Republican",
    "Under 30",
    "30-44",
    "45-64",
    "65+",
    "White",
    "Black",
    "Hispanic",
    "Male",
    "Female",
    "HS or less",
    "Some college",
    "College grad",
    "Postgrad",
]
_YOU_GOV_ROWS = {
    "Approve": "approve",
    "Disapprove": "disapprove",
    "Not sure": "not_sure",
    "Unweighted base": "unweighted_base",
    "Base": "base",
}


def _sheet_cells(xml_bytes):
    result = {}
    for row in ET.fromstring(xml_bytes).iter(_MAIN_NS + "row"):
        vals = []
        for cell in row.iter(_MAIN_NS + "c"):
            if cell.get("t") == "inlineStr":
                node = cell.find(_MAIN_NS + "is/" + _MAIN_NS + "t")
                vals.append(node.text if node is not None else "")
            else:
                node = cell.find(_MAIN_NS + "v")
                vals.append(node.text if node is not None else "")
        if vals:
            result[vals[0]] = vals[1:]
    return result


def _yougov_sheet_paths(z, workbook):
    relroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rels = {
        node.get("Id"): node for node in relroot.iter(_PACKAGE_REL_NS + "Relationship")
    }
    output, seen = [], set()
    for sheet in workbook.iter(_MAIN_NS + "sheet"):
        name, rid = sheet.get("name"), sheet.get(_REL_NS + "id")
        if not name or name in seen or rid not in rels:
            raise ValidationError(
                f"YouGov sheet has invalid name or relationship: {name!r}"
            )
        rel = rels[rid]
        if rel.get("Type") != _WORKSHEET_REL or rel.get("TargetMode") == "External":
            raise ValidationError(f"YouGov sheet {name!r} is not a local worksheet")
        target = rel.get("Target") or ""
        path = (
            posixpath.normpath(posixpath.join("xl", target))
            if not target.startswith("/")
            else posixpath.normpath(target.lstrip("/"))
        )
        if not path.startswith("xl/worksheets/") or path not in z.namelist():
            raise ValidationError(
                f"YouGov sheet {name!r} target escapes worksheet directory"
            )
        seen.add(name)
        output.append((name, path))
    if not output:
        raise ValidationError("YouGov workbook has no worksheets")
    return output


def parse_yougov_xlsx(raw: bytes, *, scored_cells=None):
    """Parse the public YouGov tracker workbook without an XLSX dependency."""
    cells = list(scored_cells or _YOU_GOV_CELLS)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            workbook = ET.fromstring(z.read("xl/workbook.xml"))
            mapped = _yougov_sheet_paths(z, workbook)
            sheets, question, dates = {}, None, None
            for name, path in mapped:
                rows = _sheet_cells(z.read(path))
                if not rows:
                    raise ValidationError(f"YouGov sheet {name!r} is empty")
                header = next(iter(rows))
                if question is None:
                    question, dates = header, rows[header]
                elif header != question or rows[header] != dates:
                    raise ValidationError(
                        f"YouGov sheet {name!r} has a different date axis"
                    )
                sheets[name] = {
                    _YOU_GOV_ROWS[k]: v for k, v in rows.items() if k in _YOU_GOV_ROWS
                }
    except (KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValidationError("YouGov response is not a valid XLSX workbook") from exc
    waves = []
    for index, day in enumerate(dates or []):
        if not day:
            continue
        wave_cells = {}
        try:
            for name in [_YOU_GOV_TOPLINE] + cells:
                rows = sheets[name]
                wave_cells[name] = {
                    "approve": round(float(rows["approve"][index]) * 100, 4),
                    "disapprove": round(float(rows["disapprove"][index]) * 100, 4),
                    "not_sure": round(float(rows["not_sure"][index]) * 100, 4),
                    "base": float(rows["base"][index]),
                    "unweighted_base": float(rows["unweighted_base"][index]),
                }
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        waves.append({"date": day, "question": question, "cells": wave_cells})
    if not waves:
        raise ValidationError("YouGov workbook produced zero complete waves")
    return waves


def yougov_package(
    raw: bytes, *, tracker: str, url: str, captured_at: str, scored_cells=None
):
    waves = parse_yougov_xlsx(raw, scored_cells=scored_cells)
    observations = []
    for wave in waves:
        observations.append(
            {
                "date": wave["date"],
                "available_at": captured_at,
                "captured_at": captured_at,
                "value": {
                    name: cell["approve"] for name, cell in wave["cells"].items()
                },
                "cells": wave["cells"],
                "source": "Economist/YouGov",
                "source_url": url,
                "content_hash": _sha256(raw),
                "provenance": "yougov_xlsx_capture",
            }
        )
    meta = {
        "source_name": "Economist/YouGov public tracker",
        "source_url": url,
        "captured_at": captured_at,
        "available_at": captured_at,
        "unit": "percentage points",
        "population": "US adults; tracker-defined cells",
        "metric": "presidential approval approve share",
        "tracker": tracker,
        "content_hash": _sha256(raw),
        "wave_count": len(waves),
    }
    content = json.dumps(
        {"tracker": tracker, "waves": waves, "metadata": meta}, ensure_ascii=False
    )
    document = _capture_document(
        title=meta["source_name"],
        url=url,
        raw=raw,
        captured_at=captured_at,
        content=content,
        observations=observations,
        candidate_items=[],
    )
    document["source_metadata"] = meta
    return {
        "manifest": meta,
        "history": observations,
        "corpus": {"synthetic": False, "documents": [document]},
    }


class YouGovRetriever:
    """Live public YouGov XLSX retriever; no API key is used."""

    def __init__(
        self,
        tracker="donald-trump-approval",
        *,
        api_base="https://api-test.yougov.com/public-data/v5/us/trackers",
        user_agent="MIT-SSA-research/0.1 (contact project maintainer)",
        timeout=60,
        transport=None,
        clock=utcnow,
        scored_cells=None,
    ):
        self.tracker = tracker
        self.api_base = api_base.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.transport = transport
        self.clock = clock
        self.scored_cells = tuple(scored_cells or _YOU_GOV_CELLS)

    def identity(self):
        return {
            "name": "yougov-public-xlsx",
            "version": 1,
            "tracker": self.tracker,
            "api_key": False,
            "scored_cells": self.scored_cells,
        }

    def supports(self, mode):
        return mode == "live"

    def search(self, query, request, *, max_results, lookback_days, timeout):
        if request.mode != "live":
            raise TemporalPolicyError(
                "YouGov live workbook cannot prove a historical snapshot"
            )
        url = _YOU_GOV_URL.format(tracker=self.tracker)
        if self.api_base != "https://api-test.yougov.com/public-data/v5/us/trackers":
            url = f"{self.api_base}/{self.tracker}/download/"
        raw = _bounded_get(
            url,
            user_agent=self.user_agent,
            timeout=min(timeout, self.timeout),
            transport=self.transport,
            max_bytes=8_000_000,
        )
        captured_at = iso(self.clock())
        package = yougov_package(
            raw,
            tracker=self.tracker,
            url=url,
            captured_at=captured_at,
            scored_cells=self.scored_cells,
        )
        return [package["corpus"]["documents"][0]]


def fetch_source_package(
    kind: str,
    *,
    tracker: str,
    output: str | Path,
    filters=None,
    user_agent="MIT-SSA-research/0.1 (contact project maintainer)",
    timeout=60,
):
    """Fetch one source and write an immutable package JSON."""
    if kind == "civiqs":
        url = f"https://civiqs.com/results/{tracker}"
        if filters:
            url += "?" + urlencode(sorted(filters.items()))
        raw = _bounded_get(url, user_agent=user_agent, timeout=timeout)
        package = civiqs_package(
            raw, tracker=tracker, url=url, captured_at=iso(utcnow()), filters=filters
        )
    elif kind == "yougov":
        url = _YOU_GOV_URL.format(tracker=tracker)
        raw = _bounded_get(
            url, user_agent=user_agent, timeout=timeout, max_bytes=8_000_000
        )
        package = yougov_package(
            raw, tracker=tracker, url=url, captured_at=iso(utcnow())
        )
    else:
        raise ValueError("source kind must be civiqs or yougov")
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing source package: {dest}")
    dest.write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return package
