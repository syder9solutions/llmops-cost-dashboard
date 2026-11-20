"""Offline regression tests for HTTP failure classification + retry policy.

Covers retry-design invariants:
- transient 429/5xx/408/409 retried with bounded backoff;
- hard quota walls (long retry_after) NOT retried;
- permanent 4xx NOT retried;
- success on a later attempt returns clean JSON;
- provider/system exceptions never leak as business validation errors.
"""
from __future__ import annotations

import json
import time
import pytest
import httpx

from src import config
from src.omniroute import OmniRouteClient, ProviderError, TransportError


class FakeResponse:
    def __init__(self, status: int, body: str = "{}") -> None:
        self.status_code = status
        self.text = body
        self._body = body

    def json(self):
        return json.loads(self._body)


class FakeHttpxClient:
    """Scripted sequence of responses/exceptions; records every post."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls = 0

    def post(self, *args, **kwargs):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def make_client(script, retries=3):
    c = OmniRouteClient(base_url="http://fake/v1", api_key="k",
                        max_retries=retries, timeout=5, min_interval_s=0)
    c._client = FakeHttpxClient(script)
    return c


def test_429_short_retry_then_success():
    body = json.dumps({"error": {"message": "rate limited", "retry_after": 0}})
    c = make_client([FakeResponse(429, body), FakeResponse(200, '{"ok": true}')])
    out = c._post("/chat/completions", {})
    assert out == {"ok": True}
    assert c._client.calls == 2


def test_5xx_retried_then_success():
    c = make_client([FakeResponse(500, "oops"), FakeResponse(503, "oops"),
                     FakeResponse(200, '{"ok": 1}')])
    assert c._post("/x", {}) == {"ok": 1}
    assert c._client.calls == 3


def test_hard_quota_wall_not_retried():
    body = json.dumps({"error": {"message": "[glm] Usage limit reached for 5 hour.",
                                 "retry_after": 4667}})
    c = make_client([FakeResponse(429, body), FakeResponse(200, '{"ok": 1}')])
    with pytest.raises(ProviderError) as exc:
        c._post("/x", {})
    assert exc.value.quota_wall is True
    assert c._client.calls == 1  # no retry into a quota wall


def test_permanent_400_not_retried():
    c = make_client([FakeResponse(400, '{"error":"bad model"}'),
                     FakeResponse(200, '{"ok": 1}')])
    with pytest.raises(ProviderError) as exc:
        c._post("/x", {})
    assert exc.value.status == 400
    assert c._client.calls == 1


def test_retry_exhaustion_raises_provider_error():
    c = make_client([FakeResponse(503, "down")], retries=2)
    with pytest.raises(ProviderError) as exc:
        c._post("/x", {})
    assert exc.value.status == 503
    assert c._client.calls == 3  # initial + 2 retries


def test_transport_error_retried():
    err = httpx.ConnectError("connection refused")
    c = make_client([err, FakeResponse(200, '{"ok": 1}')])
    assert c._post("/x", {}) == {"ok": 1}
    assert c._client.calls == 2


def test_quota_wall_detection_without_retry_after():
    e = ProviderError(429, json.dumps({"error": {"message": "quota exceeded"}}))
    assert e.quota_wall is True
    assert e.retry_after_s == 1e9


def test_retry_after_string_seconds_body(monkeypatch):
    # body with a plain numeric retry_after is parsed; small => retryable
    body = '{"error":{"message":"slow down","retry_after":0.01}}'
    e = ProviderError(429, body)
    assert e.retry_after_s == pytest.approx(0.01)
    assert e.quota_wall is False


def test_admission_throttle_enforces_min_interval():
    """Shared admission control spaces requests >= min_interval_s apart."""
    c = OmniRouteClient(base_url="http://fake/v1", api_key="k",
                        max_retries=0, min_interval_s=0.06)
    script = [FakeResponse(200, '{"ok": 1}')] * 5
    c._client = FakeHttpxClient(script)
    t0 = time.monotonic()
    for _ in range(5):
        c._post("/x", {})
    elapsed = time.monotonic() - t0
    # 5 requests -> at least 4 gaps of 0.06s
    assert elapsed >= 0.06 * 4 * 0.8, f"throttle too loose: {elapsed:.3f}s"


def test_system_error_type_distinct_from_value_error():
    # Error typing: provider/system exceptions must be ProviderError/TransportError,
    # never a bare ValueError the API layer maps to a 422 validation response.
    assert issubclass(ProviderError, RuntimeError)
    assert issubclass(TransportError, RuntimeError)
    assert not issubclass(ProviderError, ValueError)
