"""Instrumented /v1/chat/completions pipeline — the heart of the project.

Every request: trace span → cache lookup (exact→semantic) → provider call →
usage×price cost attribution → span persisted with full attrs → alerts
evaluated → response returned (OpenAI-compatible shape, plus `tracing`
metadata). Cost-aware routing: easy queries (short prompts) can be routed to
a cheaper model when `route=economy`.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from . import config
from .alerts import evaluate_alerts
from .cache import SemanticCache
from .metrics import llm_metrics
from .omniroute import (OmniRouteClient, ProviderError, TransportError,
                        chat_cost, extract_text, extract_usage)
from .pricing import chat_cost_usd
from .store import TraceStore
from .tracing import Tracer, new_trace_id


class Gateway:
    def __init__(self, store: TraceStore, client: Optional[OmniRouteClient] = None,
                 cache: Optional[SemanticCache] = None,
                 tracer: Optional[Tracer] = None,
                 alert_baseline: Optional[Dict[str, Any]] = None) -> None:
        self.store = store
        self.client = client or OmniRouteClient()
        self.cache = cache or SemanticCache()
        self.tracer = tracer or Tracer()
        self.alert_baseline = dict(alert_baseline or config.ALERT_BASELINE)
        self._sink_registered = False
        self._alert_lock = threading.Lock()
        self.total_requests = 0

    def _ensure_sink(self) -> None:
        if not self._sink_registered:
            from .tracing import add_sink
            add_sink(self._persist_span)
            self._sink_registered = True

    def _persist_span(self, span) -> None:
        try:
            self.store.add_spans([span.to_row()])
        except Exception:  # noqa: BLE001 - persistence must never break serving
            pass

    # ------------------------------------------------------------------
    def chat_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible request body → response (augmented with tracing)."""
        self._ensure_sink()
        self.total_requests += 1
        model: str = body.get("model") or config.TEXT_MODEL
        config.check_banned(model)
        messages: List[Dict[str, str]] = body.get("messages") or []
        if not messages:
            raise ValueError("messages must be a non-empty list")
        max_tokens: int = int(body.get("max_tokens") or config.MAX_TOKENS_DEFAULT)
        temperature: float = float(body.get("temperature", 0.2))
        # attribution keys (sent by workload drivers / API callers)
        feature = body.get("feature") or "default"
        user = body.get("user") or "anon"
        route = body.get("route") or "quality"   # quality | economy
        cache_control = body.get("cache", "on")  # on | off | exact-only
        trace_id = body.get("trace_id") or new_trace_id()

        # cost-aware routing: economy → cheap model for short prompts
        routed_model = model
        if route == "economy" and sum(len(str(m.get("content", ""))) for m in messages) < 600:
            routed_model = config.CHEAP_MODEL

        t_gw_start = time.perf_counter()
        with self.tracer.span("chat_completion", kind="request", trace_id=trace_id,
                              attrs={"feature": feature, "user": user,
                                     "model_requested": model,
                                     "model": routed_model, "route": route}) as req_span:
            # ---- cache lookup ------------------------------------------------
            cache_tier: Optional[str] = None
            cached_value: Optional[Dict[str, Any]] = None
            if cache_control != "off":
                semantic_on = cache_control != "exact-only"
                with self.tracer.span("cache_lookup", kind="cache",
                                      attrs={"semantic": semantic_on}):
                    self.cache.semantic_enabled = semantic_on
                    cached_value, cache_tier = self.cache.lookup(
                        routed_model, messages, max_tokens=max_tokens)

            if cached_value is not None:
                dur = (time.perf_counter() - t_gw_start) * 1000.0
                # a cache hit is itself an LLM-tier span: status='cached',
                # cost 0, duration = the (tiny) cache-serve time
                with self.tracer.span("llm_call", kind="llm_call",
                                      trace_id=trace_id,
                                      attrs={"feature": feature, "user": user,
                                             "model": routed_model,
                                             "cache_hit": cache_tier,
                                             "cost_usd": 0.0,
                                             "prompt_tokens": 0,
                                             "completion_tokens": 0,
                                             "total_tokens": 0}) as sp:
                    sp.status = "cached"
                resp = dict(cached_value)
                resp["tracing"] = {"trace_id": trace_id, "cache": cache_tier,
                                   "cost_usd": 0.0,
                                   "gateway_ms": round(dur, 2)}
                return resp

            # ---- provider call -------------------------------------------------
            with self.tracer.span("llm_call", kind="llm_call", trace_id=trace_id,
                                  attrs={"feature": feature, "user": user,
                                         "model": routed_model,
                                         "cache_hit": None}) as llm_span:
                try:
                    t0 = time.perf_counter()
                    resp = self.client.chat(
                        routed_model, messages, max_tokens=max_tokens,
                        temperature=temperature)
                    wall = time.perf_counter() - t0
                except (ProviderError, TransportError) as e:
                    llm_span.attrs["error_class"] = type(e).__name__
                    llm_span.attrs["error_detail"] = str(e)[:200]
                    raise
                usage = extract_usage(resp)
                cost, exact_price = chat_cost(routed_model, usage)
                # provider latency = wall; TTFT unknown for non-stream → use wall
                llm_span.attrs.update({
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cost_usd": round(cost, 8),
                    "price_exact": exact_price,
                    "ttft_ms": None,
                    "provider_wall_ms": round(wall * 1000.0, 2),
                })
                text = extract_text(resp)

            # ---- cache store ----------------------------------------------------
            if cache_control != "off":
                with self.tracer.span("cache_store", kind="cache"):
                    self.cache.store(routed_model, messages, resp,
                                     semantic=cache_control != "exact-only",
                                     max_tokens=max_tokens)

            # ---- response --------------------------------------------------------
            out = dict(resp)
            out["tracing"] = {
                "trace_id": trace_id, "cache": None,
                "cost_usd": round(cost, 8),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "model": routed_model,
                "feature": feature, "user": user,
            }

            # ---- alerts (cheap: only every N requests) -----------------------------
            with self._alert_lock:
                if self.total_requests % 10 == 0:
                    try:
                        evaluate_alerts(self.store, baseline=self.alert_baseline)
                    except Exception:  # noqa: BLE001
                        pass
            return out
