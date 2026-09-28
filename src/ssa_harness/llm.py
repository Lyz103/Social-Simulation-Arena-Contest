"""Bounded chat-completions HTTP transport and an explicit offline replay client."""

from dataclasses import dataclass, field
import json
import math
import os
import time
from urllib.request import Request, build_opener
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError

from .retrieval.providers import _NoRedirect
from .retrieval.utils import digest


class ModelError(RuntimeError):
    def __init__(self, message, *, transport_attempts=None):
        super().__init__(message)
        self.transport_attempts = list(transport_attempts or [])


@dataclass(frozen=True)
class ModelReply:
    text: str
    usage: dict
    model: str
    response_id: str | None = None
    transport_attempts: list[dict] = field(default_factory=list)


class ChatCompletionsClient:
    """A configured provider must implement /chat/completions.

    strict_schema=True requires that provider/model to support all generated
    schema keywords; false uses json_object plus authoritative local validation.
    truncation_retry_tokens opts into one same-prompt retry with a larger output
    limit on finish_reason=length, within the original total time budget.
    No undocumented automatic provider/model fallback is performed.
    """

    VERSION = "chat-http-v1"

    def __init__(
        self,
        *,
        model,
        endpoint,
        api_key_env="SSA_LLM_API_KEY",
        timeout=90,
        max_attempts=2,
        strict_schema=False,
        truncation_retry_tokens=None,
        transport=None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        p = urlsplit(endpoint)
        if (
            p.scheme != "https"
            or not p.hostname
            or p.username
            or p.password
            or p.query
            or p.fragment
        ):
            raise ValueError(
                "model endpoint must be an HTTPS URL without credentials/query"
            )
        if (
            not model
            or not api_key_env
            or not 1 <= timeout <= 300
            or not 1 <= max_attempts <= 3
        ):
            raise ValueError("invalid model client configuration")
        if truncation_retry_tokens is not None and (
            type(truncation_retry_tokens) is not int
            or not 1 <= truncation_retry_tokens <= 1_000_000
        ):
            raise ValueError("truncation_retry_tokens must be an integer in 1..1000000")
        self.model, self.endpoint, self.api_key_env = model, endpoint, api_key_env
        self.timeout, self.max_attempts, self.strict_schema = (
            timeout,
            max_attempts,
            strict_schema,
        )
        self.truncation_retry_tokens = truncation_retry_tokens
        self.transport = transport or build_opener(_NoRedirect()).open
        self.sleep, self.monotonic = sleep, monotonic

    def identity(self):
        return {
            "version": self.VERSION,
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
            "max_attempts": self.max_attempts,
            "strict_schema": self.strict_schema,
            "truncation_retry_tokens": self.truncation_retry_tokens,
        }

    def generate(self, compiled, *, timeout=None, repair=None):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ModelError(
                f"missing model credential environment variable: {self.api_key_env}"
            )
        messages = compiled.messages()
        if repair is not None:
            messages += [
                {"role": "assistant", "content": repair["raw"]},
                {
                    "role": "user",
                    "content": "Repair only the output format using the same evidence. Validation error: "
                    + repair["error"],
                },
            ]
        schema = compiled.output_schema
        response_format = (
            {
                "type": "json_schema",
                "json_schema": {"name": "forecast", "strict": True, "schema": schema},
            }
            if self.strict_schema
            else {"type": "json_object"}
        )
        # Schema is explicit even when only json_object mode is available.
        messages.append(
            {
                "role": "user",
                "content": "Required JSON Schema: "
                + json.dumps(schema, ensure_ascii=False),
            }
        )
        reserved = {"model", "messages", "response_format", "stream"} & set(
            compiled.model_params
        )
        if reserved:
            raise ModelError("model_params override protected protocol fields")
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
            "stream": False,
            **compiled.model_params,
        }
        end = self.monotonic() + min(
            self.timeout, timeout if timeout is not None else self.timeout
        )
        transport_attempts, token_totals = [], {}
        attempt_limit = self.max_attempts
        truncation_retried = False

        def error(message):
            return ModelError(message, transport_attempts=transport_attempts)

        # The opt-in length retry grants one extra request, not a fresh HTTP
        # retry budget. Total requests remain bounded by max_attempts + 1.
        for attempt in range(self.max_attempts + 1):
            if attempt >= attempt_limit:
                break
            started = self.monotonic()
            remaining = end - started
            if remaining <= 0:
                raise error("model time budget exhausted")
            req = Request(
                self.endpoint,
                data=json.dumps(payload, allow_nan=False).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
                method="POST",
            )
            try:
                with self.transport(req, timeout=remaining) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise error("model response too large")
                data = json.loads(raw)
                choice = data["choices"][0]
                content = choice["message"].get("content")
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage") or {}
                if not isinstance(usage, dict):
                    raise error("invalid model usage envelope")
                transport_attempts.append(
                    {
                        "finish_reason": finish_reason,
                        "model": str(data.get("model", self.model)),
                        "usage": dict(usage),
                        "content": content,
                        "id": data.get("id"),
                        "request_max_tokens": payload.get("max_tokens"),
                    }
                )
                for name, value in usage.items():
                    if (
                        isinstance(name, str)
                        and (name.endswith("_tokens") or name == "tokens")
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value >= 0
                    ):
                        token_totals[name] = token_totals.get(name, 0) + value
                if finish_reason == "length":
                    current_limit = payload.get("max_tokens")
                    if truncation_retried:
                        raise error("model response truncated (finish_reason=length) after one length retry")
                    if (
                        self.truncation_retry_tokens is None
                        or type(current_limit) is not int
                        or current_limit < 1
                        or self.truncation_retry_tokens <= current_limit
                    ):
                        raise error("model response truncated (finish_reason=length); no larger configured token retry")
                    now = self.monotonic()
                    # Do not start an evidently futile longer generation: keep
                    # at least the preceding request's duration, and one second.
                    if end - now < max(1.0, now - started):
                        raise error("model response truncated (finish_reason=length); insufficient retry time budget")
                    payload["max_tokens"] = self.truncation_retry_tokens
                    truncation_retried = True
                    attempt_limit += 1
                    continue
                if finish_reason not in (None, "stop"):
                    raise error("model response incomplete or refused")
                if not isinstance(content, str) or not content.strip():
                    raise error("model returned no text")
                return ModelReply(
                    content,
                    {**usage, **token_totals},
                    str(data.get("model", self.model)),
                    data.get("id"),
                    transport_attempts,
                )
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if (
                    status not in (429, 500, 502, 503, 504)
                    or attempt + 1 >= attempt_limit
                ):
                    raise error(f"model HTTP {status}") from None
            except (URLError, TimeoutError, OSError):
                if attempt + 1 >= attempt_limit:
                    raise error("model network error or timeout") from None
            except (ValueError, KeyError, TypeError, IndexError):
                raise error("invalid model response envelope") from None
            delay = 0.5 * 2**attempt
            if self.monotonic() + delay >= end:
                raise error("model retry budget exhausted")
            self.sleep(delay)
        raise error("model attempts exhausted")


class ReplayClient:
    """Explicit deterministic fixture client, prohibited in live pipelines."""

    synthetic = True

    def __init__(self, responses, name="offline-fixture"):
        self.responses, self.name, self.calls = dict(responses), name, 0

    def identity(self):
        return {
            "version": "replay-v1",
            "name": self.name,
            "responses_hash": digest(self.responses),
            "synthetic": True,
        }

    def generate(self, compiled, *, timeout=None, repair=None):
        self.calls += 1
        if compiled.round_id not in self.responses:
            raise ModelError("no recorded fixture response for round")
        value = self.responses[compiled.round_id]
        return ModelReply(
            value if isinstance(value, str) else json.dumps(value), {}, self.name
        )
