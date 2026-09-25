"""Bounded chat-completions HTTP transport and an explicit offline replay client."""

from dataclasses import dataclass
import json
import os
import time
from urllib.request import Request, build_opener
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError

from .retrieval.providers import _NoRedirect
from .retrieval.utils import digest


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelReply:
    text: str
    usage: dict
    model: str
    response_id: str | None = None


class ChatCompletionsClient:
    """A configured provider must implement /chat/completions.

    strict_schema=True requires that provider/model to support all generated
    schema keywords; false uses json_object plus authoritative local validation.
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
        self.model, self.endpoint, self.api_key_env = model, endpoint, api_key_env
        self.timeout, self.max_attempts, self.strict_schema = (
            timeout,
            max_attempts,
            strict_schema,
        )
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
        req = Request(
            self.endpoint,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
            method="POST",
        )
        end = self.monotonic() + min(
            self.timeout, timeout if timeout is not None else self.timeout
        )
        for attempt in range(self.max_attempts):
            remaining = end - self.monotonic()
            if remaining <= 0:
                raise ModelError("model time budget exhausted")
            try:
                with self.transport(req, timeout=remaining) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ModelError("model response too large")
                data = json.loads(raw)
                choice = data["choices"][0]
                if choice.get("finish_reason") not in (None, "stop"):
                    raise ModelError("model response incomplete or refused")
                content = choice["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ModelError("model returned no text")
                return ModelReply(
                    content,
                    data.get("usage", {}),
                    str(data.get("model", self.model)),
                    data.get("id"),
                )
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if (
                    status not in (429, 500, 502, 503, 504)
                    or attempt + 1 == self.max_attempts
                ):
                    raise ModelError(f"model HTTP {status}") from None
            except (URLError, TimeoutError, OSError):
                if attempt + 1 == self.max_attempts:
                    raise ModelError("model network error or timeout") from None
            except (ValueError, KeyError, TypeError, IndexError):
                raise ModelError("invalid model response envelope") from None
            delay = 0.5 * 2**attempt
            if self.monotonic() + delay >= end:
                raise ModelError("model retry budget exhausted")
            self.sleep(delay)
        raise ModelError("model attempts exhausted")


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
