"""JSON, timestamps and text utilities; no network access."""

import hashlib
import json
import re
from datetime import datetime, time, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .errors import ValidationError


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utcnow():
    return datetime.now(timezone.utc)


def iso(value):
    if value.tzinfo is None:
        raise ValidationError("时间必须携带时区，例如 Z 或 +08:00")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_instant(value):
    if not isinstance(value, str):
        raise ValidationError("时间必须是带时区的 ISO 8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("时间不是有效的 ISO 8601 格式") from exc
    if parsed.tzinfo is None:
        raise ValidationError("时间缺少时区")
    return parsed.astimezone(timezone.utc)


def source_time_upper(value):
    """Date-only metadata is conservatively represented by end-of-day UTC.

    Metadata does not prove when bytes became public. Strict snapshot mode
    separately checks the captured bytes' timestamp.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("来源缺少日期")
    value = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            day = datetime.strptime(value, "%Y-%m-%d").date()
            return datetime.combine(day, time.max, tzinfo=timezone.utc)
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            result = parsedate_to_datetime(value)
        if result.tzinfo is None:
            raise ValueError("missing timezone")
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("无法解析来源日期或缺少时区") from exc


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        elif tag in ("p", "div", "br", "li", "td", "tr"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1
        elif tag in ("p", "div", "li", "td", "tr"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    if not isinstance(value, str):
        return ""
    parser = _TextParser()
    parser.feed(value)
    # Remove invisible formatting characters, but preserve ordinary Unicode.
    text = "".join(parser.parts)
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]",
        "",
        text,
    )
    return " ".join(text.split())


def canonical_url(value):
    """Remove known trackers only; id, date, q and other semantic params stay."""
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValidationError("无效来源 URL")
    try:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("scheme or host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("credentials in URL")
        port = parts.port
        host = parts.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if port and not (
            (parts.scheme == "https" and port == 443)
            or (parts.scheme == "http" and port == 80)
        ):
            host += f":{port}"
        query = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")
            and k.lower() not in ("fbclid", "gclid", "msclkid")
        ]
        return urlunsplit((parts.scheme, host, parts.path or "/", urlencode(query), ""))
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("无效来源 URL") from exc


def domain_matches(host, allowed):
    return any(host == domain or host.endswith("." + domain) for domain in allowed)
