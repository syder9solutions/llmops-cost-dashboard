"""Alert rules over the trace store. Each rule: evaluate(store, window) →
{firing, reason, metrics}. Deterministic, testable, no timers — the API layer
calls evaluate_alerts() after each request (and scripts replay windows)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from .metrics import llm_metrics
from .store import TraceStore

RULES_VERSION = "1.0"


def psi(expected: List[float], actual: List[float], nbins: int = 10) -> Optional[float]:
    """Population Stability Index over output-token distributions.

    Quantile-bin on the expected sample so bins are population-proportional;
    PSI = Σ (a_i - e_i) · ln(a_i / e_i). Deterministic, order-free.
    """
    if len(expected) < 80 or len(actual) < 80:
        return None   # PSI is too noisy below ~80 samples per side
    es, as_ = sorted(expected), sorted(actual)

    def quantile(s: List[float], q: float) -> float:
        k = max(1, math.ceil(q * len(s)))
        return s[min(k, len(s)) - 1]

    edges = [quantile(es, i / nbins) for i in range(1, nbins)]
    if not edges or edges[-1] <= edges[0]:
        return None

    def hist(s: List[float]) -> List[float]:
        h = [0] * nbins
        for v in s:
            b = 0
            for e in edges:
                if v > e:
                    b += 1
                else:
                    break
            h[b] += 1
        return [max(c / len(s), 1e-6) for c in h]  # floor avoids div/0

    he, ha = hist(es), hist(as_)
    return sum((a - e) * math.log(a / e) for a, e in zip(ha, he))


def _span_out_tokens(sp: Dict) -> Optional[float]:
    v = sp.get("attrs", {}).get("completion_tokens")
    return float(v) if v is not None else None


def evaluate_alerts(store: TraceStore, now_ts: Optional[float] = None,
                    window_min: float = 10.0, baseline: Optional[Dict] = None
                    ) -> Dict[str, Any]:
    """Evaluate all rules over the recent window; returns per-rule results."""
    baseline = baseline or {}
    spans = store.llm_spans(since_ts=(now_ts - window_min * 60) if now_ts else None)
    m = llm_metrics(spans)["*"] if spans else None
    out: Dict[str, Any] = {"rules": {}, "evaluated_at": now_ts,
                           "window_min": window_min, "requests_in_window": len(spans)}

    n = len(spans)
    # --- error-rate jump ------------------------------------------------
    if n >= 10:
        errs = sum(1 for s in spans if s.get("status") == "error")
        rate = errs / n
        thr = baseline.get("error_rate", 0.10)
        out["rules"]["error_rate_jump"] = {
            "firing": rate > thr,
            "reason": f"error rate {rate:.1%} over last {n} calls (threshold {thr:.0%})"
                      if rate > thr else f"error rate {rate:.1%} ok",
            "value": rate, "threshold": thr,
        }

    # --- p95 latency drift ----------------------------------------------
    lats = [s["dur_ms"] for s in spans
            if s.get("status") == "ok" and s.get("dur_ms") is not None]
    if len(lats) >= 10:
        from .metrics import p95 as _p95
        cur = _p95(lats)
        base = baseline.get("p95_ms", 3000.0)
        ratio_thr = baseline.get("p95_ratio", 1.5)
        firing = cur is not None and cur > base * ratio_thr
        out["rules"]["p95_latency_drift"] = {
            "firing": firing,
            "reason": (f"p95 {cur:.0f}ms > {ratio_thr}× baseline "
                       f"{base:.0f}ms") if firing else f"p95 {cur:.0f}ms ok",
            "value": cur, "threshold": base * ratio_thr,
        }

    # --- cost per 1k spike ----------------------------------------------
    if n >= 10:
        costs = [s.get("attrs", {}).get("cost_usd", 0.0) for s in spans]
        per_1k = sum(costs) / n * 1000.0
        base = baseline.get("cost_per_1k", None)
        if base:
            ratio_thr = baseline.get("cost_ratio", 1.5)
            firing = per_1k > base * ratio_thr
            out["rules"]["cost_spike"] = {
                "firing": firing,
                "reason": (f"${per_1k:.3f}/1k > {ratio_thr}× baseline ${base:.3f}"
                           if firing else f"${per_1k:.3f}/1k ok"),
                "value": per_1k, "threshold": base * ratio_thr,
            }

    # --- output-token distribution drift (PSI) ---------------------------
    # drift measures SUCCESSFUL output distribution; error spans have no
    # completion output in reality and would contaminate the sample
    outs_actual = [v for v in (
        _span_out_tokens(s) for s in spans if s.get("status") == "ok")
        if v is not None]
    expected = baseline.get("out_tokens_sample")
    if expected and len(outs_actual) >= 20:
        val = psi(list(expected), outs_actual)
        if val is not None:
            thr = baseline.get("psi_threshold", 0.25)
            out["rules"]["output_drift"] = {
                "firing": val > thr,
                "reason": f"PSI {val:.3f} {'>' if val > thr else '<='} {thr}",
                "value": val, "threshold": thr,
            }

    # --- cache hit-rate collapse -----------------------------------------
    # armed only when the deployment expects caching (cache_expected=True):
    # a cache-off baseline phase would otherwise false-alarm on 0% hits.
    cached = sum(1 for s in spans if s.get("status") == "cached")
    if n >= 20 and baseline.get("cache_expected"):
        hr = cached / n
        floor = baseline.get("min_hit_rate", 0.05)
        out["rules"]["cache_hit_rate_collapse"] = {
            "firing": hr < floor,
            "reason": f"hit-rate {hr:.1%} {'<' if hr < floor else '>='} floor {floor:.0%}",
            "value": hr, "threshold": floor,
        }

    # --- budget burn (per-user/day) ---------------------------------------
    users: Dict[str, float] = {}
    for s in spans:
        u = s.get("attrs", {}).get("user")
        if u is not None:
            users[u] = users.get(u, 0.0) + s.get("attrs", {}).get("cost_usd", 0.0)
    budget = baseline.get("user_daily_budget")
    if budget:
        burn = {u: c for u, c in users.items() if c > budget}
        out["rules"]["budget_burn"] = {
            "firing": bool(burn),
            "reason": (f"users over budget: {sorted(burn)}" if burn
                       else "all users within budget"),
            "value": max(users.values()) if users else 0.0, "threshold": budget,
        }

    out["firing"] = {k: v["firing"] for k, v in out["rules"].items()}
    out["any_firing"] = any(out["firing"].values())
    return out
