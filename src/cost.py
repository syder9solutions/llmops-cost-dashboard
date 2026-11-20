"""Cost attribution rollups: request → feature → user → model → day.

Reads the trace store (llm spans) and produces the "$ per X" tables the
dashboard and README headline table are built from.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .store import TraceStore


def _day(ts: float) -> str:
    import time
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def attribution(spans: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up cost by feature / user / model / day. Cached hits cost $0 and
    are counted separately (that IS the savings story)."""
    by_feature: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"cost": 0.0, "requests": 0, "cached": 0, "errors": 0,
                 "tokens_in": 0, "tokens_out": 0})
    by_user: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"cost": 0.0, "requests": 0, "cached": 0})
    by_model: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"cost": 0.0, "requests": 0, "cached": 0, "tokens_in": 0,
                 "tokens_out": 0})
    by_day: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"cost": 0.0, "requests": 0, "cached": 0})

    for sp in spans:
        a = sp.get("attrs", {})
        cost = float(a.get("cost_usd", 0.0) or 0.0)
        cached = sp.get("status") == "cached"
        err = sp.get("status") == "error"
        feature = str(a.get("feature", "default"))
        user = str(a.get("user", "anon"))
        model = str(a.get("model", "?"))
        day = _day(sp.get("ts", 0.0))
        tin = int(a.get("prompt_tokens", 0) or 0)
        tout = int(a.get("completion_tokens", 0) or 0)

        for tbl, key in ((by_feature, feature), (by_user, user),
                         (by_model, model), (by_day, day)):
            tbl[key]["cost"] += cost
            tbl[key]["requests"] += 1
            if cached:
                tbl[key]["cached"] += 1

        by_feature[feature]["errors"] += 1 if err else 0
        by_feature[feature]["tokens_in"] += tin
        by_feature[feature]["tokens_out"] += tout
        by_model[model]["tokens_in"] += tin
        by_model[model]["tokens_out"] += tout

    def _round(d: Dict) -> Dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in d.items()}

    return {
        "by_feature": {k: _round(v) for k, v in sorted(by_feature.items())},
        "by_user": {k: _round(v) for k, v in sorted(by_user.items())},
        "by_model": {k: _round(v) for k, v in sorted(by_model.items())},
        "by_day": {k: _round(v) for k, v in sorted(by_day.items())},
        "total_cost_usd": round(sum(v["cost"] for v in by_feature.values()), 6),
        "total_requests": sum(int(v["requests"]) for v in by_feature.values()),
        "total_cached": sum(int(v["cached"]) for v in by_feature.values()),
    }


def top_cost_features(spans: Iterable[Dict[str, Any]], n: int = 10) -> List:
    a = attribution(spans)
    feats = sorted(a["by_feature"].items(), key=lambda kv: -kv[1]["cost"])
    return [{"feature": k, **v} for k, v in feats[:n]]


def attribution_from_store(store: TraceStore,
                           since_ts: Optional[float] = None) -> Dict[str, Any]:
    return attribution(store.llm_spans(since_ts=since_ts))
