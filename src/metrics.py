"""Latency percentiles (nearest-rank), rates and aggregation helpers."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from .store import TraceStore


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile. p in (0,100]. Empty → None."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(p / 100.0 * len(s)))
    return s[k - 1]


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def p50(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 50)


def p95(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 95)


def p99(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 99)


def rates(parts: Dict[str, int]) -> Dict[str, float]:
    total = sum(parts.values())
    if total == 0:
        return {k: 0.0 for k in parts}
    return {k: v / total for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Aggregations over llm spans (source of truth = trace store)
# ---------------------------------------------------------------------------

def llm_metrics(spans: Iterable[Dict], bucket_key: Optional[str] = None
                ) -> Dict[str, Dict[str, object]]:
    """Per-bucket (default: single '*') latency/cost/token/cache stats.

    Bucket key examples: attrs.feature, attrs.user.
    Cached calls (status='cached') count toward hit-rate but are excluded
    from latency percentiles of live model calls (they have their own metric).
    """
    groups: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"lat": [], "lat_cached": [], "cost": [], "in_tok": [],
                 "out_tok": [], "errors": 0, "calls": 0, "cached": 0})
    for sp in spans:
        a = sp.get("attrs", {})
        key = (a.get(bucket_key) if bucket_key else None) or "*"
        g = groups[key]
        if sp.get("status") == "cached":
            g["cached"] += 1
            g["lat_cached"].append(sp.get("dur_ms", 0.0))
            continue
        g["calls"] += 1
        if sp.get("status") == "error":
            g["errors"] += 1
            continue
        g["lat"].append(sp.get("dur_ms", 0.0))
        g["cost"].append(a.get("cost_usd", 0.0))
        g["in_tok"].append(a.get("prompt_tokens", 0))
        g["out_tok"].append(a.get("completion_tokens", 0))

    out: Dict[str, Dict[str, object]] = {}
    for key, g in groups.items():
        total_calls = g["calls"] + g["cached"]
        out[key] = {
            "requests": total_calls,
            "live_calls": g["calls"],
            "errors": g["errors"],
            "cached": g["cached"],
            "cache_hit_rate": (g["cached"] / total_calls) if total_calls else 0.0,
            "error_rate": (g["errors"] / g["calls"]) if g["calls"] else 0.0,
            "latency_p50_ms": p50(g["lat"]),
            "latency_p95_ms": p95(g["lat"]),
            "latency_p99_ms": p99(g["lat"]),
            "latency_mean_ms": mean(g["lat"]),
            "cached_latency_p50_ms": p50(g["lat_cached"]),
            "cost_usd": sum(g["cost"]),
            "prompt_tokens": sum(g["in_tok"]),
            "completion_tokens": sum(g["out_tok"]),
            "cost_per_1k_requests": (sum(g["cost"]) / total_calls * 1000.0)
                                    if total_calls else 0.0,
        }
    return dict(out)


def metrics_from_store(store: TraceStore, since_ts: Optional[float] = None,
                       bucket_key: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    return llm_metrics(store.llm_spans(since_ts=since_ts), bucket_key=bucket_key)


def time_buckets(spans: Iterable[Dict], bucket_s: float = 60.0
                 ) -> List[Dict[str, object]]:
    """Time-series rows for dashboard charts: per time bucket totals."""
    rows: Dict[float, Dict[str, float]] = {}
    for sp in spans:
        a = sp.get("attrs", {})
        b = math.floor(sp["ts"] / bucket_s) * bucket_s
        r = rows.setdefault(b, {"reqs": 0, "cost": 0.0, "lat_sum": 0.0,
                                "lat_n": 0, "errors": 0, "cached": 0})
        r["reqs"] += 1
        if sp.get("status") == "cached":
            r["cached"] += 1
        elif sp.get("status") == "error":
            r["errors"] += 1
        else:
            r["cost"] += a.get("cost_usd", 0.0)
            r["lat_sum"] += sp.get("dur_ms", 0.0)
            r["lat_n"] += 1
    return [{"ts": k, "reqs": v["reqs"], "cost_usd": round(v["cost"], 6),
             "latency_mean_ms": (v["lat_sum"] / v["lat_n"]) if v["lat_n"] else None,
             "errors": v["errors"], "cached": v["cached"]}
            for k, v in sorted(rows.items())]
