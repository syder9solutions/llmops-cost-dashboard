"""OpenAI-compatible httpx client for the OmniRoute gateway.

Failure classification (see retry-design):
- transport errors + HTTP 5xx + HTTP 408/409/429 -> transient, retried with
  bounded exponential backoff + jitter, capped by a per-call deadline;
- HTTP 4xx other than 408/409/429 -> permanent, raised immediately;
- a 429 whose body reports a long `retry_after`/quota-wall reset (>> our
  deadline) is NOT retried: waiting could not help within budget and hammering
  a quota-walled model only wastes requests.

Returns raw JSON dicts (usage preserved verbatim).
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from . import config
from .pricing import chat_cost_usd, embed_cost_usd


class ProviderError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"provider {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.retry_after_s: Optional[float] = None
        self.quota_wall = False
        # Surface an explicit retry-after (seconds, or "29s"/"1m") from body
        try:
            payload = json.loads(body)
            err = payload.get("error", payload)
            raw = err.get("retry_after") if isinstance(err, dict) else None
            if raw is not None:
                self.retry_after_s = float(raw)
            else:
                msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
                m = re.search(r"reset after\s*(\d+)\s*(m|s)", msg, re.I)
                if m:
                    self.retry_after_s = (int(m.group(1)) * 60 if m.group(2).lower() == "m"
                                          else int(m.group(1)))
                elif "usage limit reached" in msg.lower() or "quota" in msg.lower():
                    # no parseable delay -> treat as a long wall
                    self.retry_after_s = 1e9
        except (ValueError, TypeError):
            pass
        if self.status == 429 and (self.retry_after_s or 0) > config.MAX_RETRY_WAIT_S:
            self.quota_wall = True


class TransportError(RuntimeError):
    pass


class OmniRouteClient:
    # shared admission control: throttle ALL instances so a concurrent workload
    # cannot trip the provider's per-org rate limit with its own burst.
    _throttle_lock = threading.Lock()
    _last_call_mono = 0.0

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: Optional[float] = None,
                 max_retries: Optional[int] = None,
                 min_interval_s: Optional[float] = None) -> None:
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.timeout = timeout or config.REQUEST_TIMEOUT_S
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.min_interval_s = (config.MIN_CALL_INTERVAL_S if min_interval_s is None
                               else min_interval_s)
        self._client = httpx.Client(timeout=self.timeout)

    def _admit(self) -> None:
        """Wait until the shared min-interval since the last request elapses."""
        if self.min_interval_s <= 0:
            return
        while True:
            with OmniRouteClient._throttle_lock:
                now = time.monotonic()
                wait = OmniRouteClient._last_call_mono + self.min_interval_s - now
                if wait <= 0:
                    OmniRouteClient._last_call_mono = now
                    return
            time.sleep(wait)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last: Optional[Exception] = None
        deadline = time.monotonic() + config.MAX_RETRY_WAIT_S
        for attempt in range(self.max_retries + 1):
            self._admit()  # admission control before each attempt (incl. retries)
            try:
                r = self._client.post(url, headers=self._headers(), json=payload)
            except httpx.HTTPError as e:
                last = TransportError(f"{type(e).__name__}: {e}")
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0) * (0.5 + random.random()))
                    continue
                raise last
            if r.status_code == 200:
                return r.json()
            if r.status_code >= 500 or r.status_code in (408, 409, 429):
                err: Exception
                if r.status_code == 429:
                    err = ProviderError(r.status_code, r.text)
                    # hard quota wall: retrying within budget cannot help
                    if getattr(err, "quota_wall", False):
                        raise err
                else:
                    err = ProviderError(r.status_code, r.text)
                last = err
                if attempt < self.max_retries and time.monotonic() < deadline:
                    ra = getattr(err, "retry_after_s", None) or 0
                    delay = max(0.5 * (2 ** attempt), float(ra))
                    delay = min(delay, config.MAX_RETRY_WAIT_S)
                    if time.monotonic() + delay >= deadline:
                        raise err
                    time.sleep(delay * (0.75 + 0.5 * random.random()))
                    continue
                raise err
            # all other 4xx are permanent: no retry
            raise ProviderError(r.status_code, r.text)
        raise last or TransportError("unreachable")

    # -- chat -------------------------------------------------------------
    def chat(self, model: str, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None, temperature: float = 0.2,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config.check_banned(model)
        payload: Dict[str, Any] = {"model": model, "messages": messages,
                                   "max_tokens": max_tokens if max_tokens is not None
                                   else config.MAX_TOKENS_DEFAULT,
                                   "temperature": temperature}
        if extra:
            payload.update(extra)
        return self._post("/chat/completions", payload)

    # -- embeddings ---------------------------------------------------------
    def embed(self, model: str, inputs: List[str]) -> Dict[str, Any]:
        config.check_banned(model)
        return self._post("/embeddings", {"model": model, "input": inputs})

    def close(self) -> None:
        self._client.close()


def extract_usage(resp: Dict[str, Any]) -> Dict[str, int]:
    u = resp.get("usage") or {}
    return {"prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0)}


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def chat_cost(model: str, usage: Dict[str, int]) -> tuple:
    return chat_cost_usd(model, usage["prompt_tokens"], usage["completion_tokens"])


def embed_cost(model: str, usage: Dict[str, int]) -> tuple:
    return embed_cost_usd(model, usage.get("total_tokens", 0))
