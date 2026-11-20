"""Shared live-run driver: executes a workload against the Gateway (in-process)
with concurrency, collects per-request outcomes, and writes a JSON summary.

Runs against the REAL gateway (live provider calls). Use --dry-run for a
synthetic-latency mode that skips network (smoke-testing the harness only).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.applogic import Gateway  # noqa: E402
from src.omniroute import OmniRouteClient  # noqa: E402
from src.store import TraceStore  # noqa: E402


class DryClient:
    """Offline stand-in: deterministic latencies + fixed usage. NO network."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model, messages, max_tokens=None, temperature=0.2, extra=None):
        self.calls += 1
        time.sleep(0.05 + (len(str(messages)) % 50) / 1000.0)
        ptok = 40 + len(str(messages)) // 8
        ctok = 20 + (self.calls * 7) % 90
        return {
            "id": f"dry-{self.calls}", "model": model,
            "choices": [{"message": {"role": "assistant",
                         "content": f"dry answer #{self.calls}"}}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                      "total_tokens": ptok + ctok},
        }

    def close(self):  # noqa: D102
        pass


def run_workload(gw: Gateway, workload: List[Dict[str, Any]], workers: int = 4,
                 progress_every: int = 50) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def one(item: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = gw.chat_completion({
                "model": config.TEXT_MODEL,
                "messages": item["messages"],
                "max_tokens": item.get("max_tokens", 120),
                "feature": item["feature"], "user": item["user"],
                "route": item.get("route", "quality"),
                "cache": item.get("cache", "on"),
            })
            wall = (time.perf_counter() - t0) * 1000.0
            tr = resp.get("tracing", {})
            return {"i": item["i"], "ok": True, "wall_ms": wall,
                    "cache": tr.get("cache"), "cost_usd": tr.get("cost_usd", 0.0),
                    "model": tr.get("model"),
                    "prompt_tokens": tr.get("prompt_tokens"),
                    "completion_tokens": tr.get("completion_tokens")}
        except Exception as e:  # noqa: BLE001
            return {"i": item["i"], "ok": False, "wall_ms": (time.perf_counter() - t0) * 1000,
                    "error": f"{type(e).__name__}: {e}"}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(one, workload), 1):
            results.append(r)
            if progress_every and k % progress_every == 0:
                print(f"  {k}/{len(workload)} done ({time.time()-t_start:.0f}s)",
                      flush=True)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    live = [r for r in ok if r.get("cache") is None]
    cached = [r for r in ok if r.get("cache")]
    walls = sorted(r["wall_ms"] for r in ok)
    cost = sum(r.get("cost_usd", 0.0) for r in ok)
    n = len(results)
    import math

    def pct(v, p):
        if not v:
            return None
        k = max(1, math.ceil(p / 100 * len(v)))
        return v[k - 1]

    return {
        "requests": n, "errors": len(errs),
        "cache_hits": len(cached),
        "hit_rate": round(len(cached) / n, 4) if n else 0.0,
        "exact_hits": sum(1 for r in cached if r.get("cache") == "exact"),
        "semantic_hits": sum(1 for r in cached if r.get("cache") == "semantic"),
        "cost_usd_total": round(cost, 6),
        "cost_per_1k": round(cost / n * 1000.0, 6) if n else 0.0,
        "p50_ms": pct(walls, 50), "p95_ms": pct(walls, 95),
        "mean_ms": round(sum(walls) / len(walls), 1) if walls else None,
        "tokens_in": sum(r.get("prompt_tokens") or 0 for r in ok),
        "tokens_out": sum(r.get("completion_tokens") or 0 for r in ok),
        "live_calls": len(live),
    }


def make_gateway(dry: bool = False, cache_mode: str = "on") -> Gateway:
    store = TraceStore(config.TRACE_DB)
    client: Any = DryClient() if dry else OmniRouteClient()
    from src.cache import SemanticCache
    sem_enabled = cache_mode != "exact-only"
    cache = SemanticCache(semantic_enabled=sem_enabled)
    return Gateway(store, client=client, cache=cache)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(f"wrote {path}")
