"""Bounded, auditable retries for a provider's explicit length truncation."""

from dataclasses import asdict
import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from ssa_harness.llm import ChatCompletionsClient, ModelError, ModelReply


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def response(reason="stop", *, usage=None, content='{"mean":1,"sd":2}', id="reply"):
    return {
        "id": id,
        "model": "same-model",
        "choices": [{"finish_reason": reason, "message": {"content": content}}],
        "usage": usage or {},
    }


def client(monkeypatch, replies, *, durations=(), **kwargs):
    monkeypatch.setenv("SSA_TRUNCATION_TEST_KEY", "test-only-token")
    clock = FakeClock()
    calls = []
    results = iter(replies)

    def transport(req, timeout):
        calls.append({"payload": json.loads(req.data), "timeout": timeout})
        index = len(calls) - 1
        clock.now += durations[index] if index < len(durations) else 0
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    result = ChatCompletionsClient(
        model="same-model",
        endpoint="https://example.test/chat/completions",
        api_key_env="SSA_TRUNCATION_TEST_KEY",
        transport=transport,
        monotonic=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return result, calls


def compiled(max_tokens=16):
    return SimpleNamespace(
        messages=lambda: [{"role": "user", "content": "Forecast next value"}],
        output_schema={"type": "object"},
        model_params={"max_tokens": max_tokens, "temperature": 0},
    )


def test_retry_increases_only_token_limit_and_accumulates_all_top_level_usage(monkeypatch):
    first_usage = {
        "prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26,
        "prompt_cache_hit_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 15},
    }
    final_usage = {
        "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
        "prompt_cache_hit_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 12},
    }
    runner, calls = client(monkeypatch, [
        response("length", usage=first_usage, content='{"mean":', id="truncated"),
        response(usage=final_usage, id="complete"),
    ], durations=[4, 3], timeout=20, max_attempts=1, truncation_retry_tokens=32)
    prompt = compiled()
    reply = runner.generate(prompt)
    assert len(calls) == 2
    assert calls[0]["payload"]["max_tokens"] == 16
    assert calls[1]["payload"] == {**calls[0]["payload"], "max_tokens": 32}
    assert calls[1]["timeout"] == 16
    assert prompt.model_params["max_tokens"] == 16
    assert reply.usage["prompt_tokens"] == 20
    assert reply.usage["completion_tokens"] == 36
    assert reply.usage["total_tokens"] == 56
    assert reply.usage["prompt_cache_hit_tokens"] == 18
    assert reply.response_id == "complete"
    assert reply.model == "same-model"
    assert reply.transport_attempts == [
        {"finish_reason": "length", "model": "same-model", "usage": first_usage,
         "content": '{"mean":', "id": "truncated", "request_max_tokens": 16},
        {"finish_reason": "stop", "model": "same-model", "usage": final_usage,
         "content": '{"mean":1,"sd":2}', "id": "complete", "request_max_tokens": 32},
    ]
    assert "test-only-token" not in json.dumps(asdict(reply))
    assert "Authorization" not in json.dumps(asdict(reply))


@pytest.mark.parametrize("retry_tokens", [None, 8, 16])
def test_disabled_or_not_larger_limit_never_retries(monkeypatch, retry_tokens):
    runner, calls = client(monkeypatch, [response("length")], truncation_retry_tokens=retry_tokens)
    with pytest.raises(ModelError, match="no larger configured token retry") as error:
        runner.generate(compiled())
    assert len(calls) == 1
    assert error.value.transport_attempts[0]["finish_reason"] == "length"


def test_insufficient_remaining_total_budget_does_not_start_retry(monkeypatch):
    runner, calls = client(monkeypatch, [response("length")],
                           durations=[6], timeout=20, truncation_retry_tokens=32)
    with pytest.raises(ModelError, match="insufficient retry time budget"):
        runner.generate(compiled(), timeout=10)
    assert len(calls) == 1
    assert calls[0]["timeout"] == 10


@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "error"])
def test_non_length_incomplete_or_refused_response_does_not_retry(monkeypatch, reason):
    runner, calls = client(monkeypatch, [response(reason)], truncation_retry_tokens=32)
    with pytest.raises(ModelError, match="incomplete or refused") as error:
        runner.generate(compiled())
    assert len(calls) == 1
    assert error.value.transport_attempts[0]["finish_reason"] == reason


def test_second_length_response_fails_explicitly_and_retains_both_attempts(monkeypatch):
    runner, calls = client(monkeypatch, [response("length"), response("length")],
                           truncation_retry_tokens=32)
    with pytest.raises(ModelError, match="after one length retry") as error:
        runner.generate(compiled())
    assert len(calls) == 2
    assert len(error.value.transport_attempts) == 2
    assert [a["request_max_tokens"] for a in error.value.transport_attempts] == [16, 32]


@pytest.mark.parametrize("partial_content", [None, ""])
def test_length_retry_allows_empty_partial_content(monkeypatch, partial_content):
    runner, calls = client(monkeypatch, [response("length", content=partial_content), response()],
                           max_attempts=1, truncation_retry_tokens=32)
    reply = runner.generate(compiled())
    assert len(calls) == 2
    assert reply.transport_attempts[0]["content"] == partial_content
    assert reply.text == '{"mean":1,"sd":2}'


def test_http_retries_share_bounded_budget_with_one_extra_length_request(monkeypatch):
    unavailable = HTTPError("https://example.test", 503, "unavailable", {}, None)
    runner, calls = client(monkeypatch, [unavailable, response("length"), response()],
                           max_attempts=2, truncation_retry_tokens=32)
    reply = runner.generate(compiled())
    assert len(calls) == 3
    assert [call["payload"]["max_tokens"] for call in calls] == [16, 16, 32]
    assert len(reply.transport_attempts) == 2


def test_http_failures_after_length_retry_cannot_restart_retry_budget(monkeypatch):
    failures = [HTTPError("https://example.test", 503, "unavailable", {}, None) for _ in range(2)]
    runner, calls = client(monkeypatch, [response("length"), *failures],
                           max_attempts=2, truncation_retry_tokens=32)
    with pytest.raises(ModelError, match="HTTP 503") as error:
        runner.generate(compiled())
    assert len(calls) == 3
    assert len(error.value.transport_attempts) == 1


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "32", 1_000_001])
def test_invalid_retry_configuration_rejected(value):
    with pytest.raises(ValueError, match="truncation_retry_tokens"):
        ChatCompletionsClient(model="same-model", endpoint="https://example.test", truncation_retry_tokens=value)


def test_retry_configuration_changes_identity_and_reply_constructor_is_compatible(monkeypatch):
    old, _ = client(monkeypatch, [], truncation_retry_tokens=None)
    new, _ = client(monkeypatch, [], truncation_retry_tokens=32)
    assert old.identity() != new.identity()
    assert new.identity()["truncation_retry_tokens"] == 32
    assert ModelReply("{}", {}, "fixture").transport_attempts == []
