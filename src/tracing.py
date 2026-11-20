"""Lightweight span/trace primitives (OpenTelemetry GenAI-style, zero deps).

Every unit of work the gateway performs becomes a span with:
  trace_id  — groups a whole request (cache check, model call, DB write…)
  span_id   — unique within the trace
  parent_id — nesting (stage spans under the request span)
  kind      — request | llm_call | cache | embed | store | internal
  status    — ok | error | cached

Timing: monotonic start/end → wall-clock `ts` + `dur_ms`. LLM spans carry
usage/cost/model/cache attributes per OTel GenAI semantic conventions.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

_SPAN_LOCK = threading.Lock()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    kind: str = "internal"  # request|llm_call|cache|embed|store|internal
    parent_id: Optional[str] = None
    ts: float = 0.0          # unix seconds at start
    start_monotonic: float = 0.0
    dur_ms: float = 0.0
    status: str = "ok"       # ok|error|cached
    error_type: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_id": self.parent_id, "name": self.name, "kind": self.kind,
            "ts": self.ts, "dur_ms": self.dur_ms, "status": self.status,
            "error_type": self.error_type, "attrs": self.attrs,
        }


class Tracer:
    """Thread-safe span factory. Span objects stay plain data; the store owns
    persistence. `extra` on __exit__ merges final attributes (usage, cost…)."""

    def __init__(self) -> None:
        self._stack = threading.local()

    # -- context stack -------------------------------------------------
    def _stack_list(self) -> list:
        s = getattr(self._stack, "stack", None)
        if s is None:
            s = []
            self._stack.stack = s
        return s

    @contextmanager
    def span(self, name: str, kind: str = "internal",
             trace_id: Optional[str] = None, attrs: Optional[Dict[str, Any]] = None
             ) -> Iterator[Span]:
        stack = self._stack_list()
        parent = stack[-1] if stack else None
        sp = Span(
            trace_id=trace_id or (parent.trace_id if parent else new_trace_id()),
            span_id=new_span_id(),
            name=name, kind=kind,
            parent_id=parent.span_id if parent else None,
            ts=time.time(), start_monotonic=time.perf_counter(),
            attrs=dict(attrs or {}),
        )
        stack.append(sp)
        # instrumentation overhead = span-machinery cost measured BEFORE the
        # wrapped work runs (context setup + object creation beyond user time)
        t_overhead = time.perf_counter() - sp.start_monotonic
        try:
            yield sp
        except BaseException as exc:  # noqa: BLE001 - span must observe all errors
            sp.status = "error"
            sp.error_type = type(exc).__name__
            raise
        finally:
            sp.dur_ms = (time.perf_counter() - sp.start_monotonic) * 1000.0
            stack.pop()
            sp.attrs.setdefault("overhead_ms", round(t_overhead * 1000.0, 3))
            _SPAN_LOCK.acquire()
            try:
                for sink in _SINKS:
                    sink(sp)
            finally:
                _SPAN_LOCK.release()


# module-level sinks: the store registers itself; tests can add capturers
_SINKS: list = []


def add_sink(fn) -> None:
    _SINKS.append(fn)


def clear_sinks() -> None:
    _SINKS.clear()


def now_iso(ts: Optional[float] = None) -> str:
    t = ts if ts is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + "Z"
