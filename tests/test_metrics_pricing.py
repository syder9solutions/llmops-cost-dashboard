"""Unit tests: percentiles, metrics rollups, attribution, pricing."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (llm_metrics, mean, p50, p95, percentile,  # noqa: E402
                         time_buckets)
from src.pricing import (PRICES_VERSION, chat_cost_usd, embed_cost_usd,  # noqa: E402
                         price_info, resolve)
from src.cost import attribution  # noqa: E402


# ---------------- percentiles -------------------------------------------------
def test_percentile_basic():
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    assert percentile(v, 50) == 5   # nearest-rank: ceil(0.5*10)=5th → 5
    assert percentile(v, 95) == 100  # ceil(9.5)=10th
    assert percentile(v, 100) == 100


def test_percentile_empty():
    assert percentile([], 50) is None
    assert mean([]) is None


def test_percentile_single():
    assert percentile([7], 50) == 7
    assert percentile([7], 95) == 7


def test_p50_p95_helpers():
    v = list(range(1, 101))
    assert p50(v) == 50
    assert p95(v) == 95


# ---------------- pricing -----------------------------------------------------
def test_known_price():
    # glm/glm-5.3-flash: $0.10 in / $0.40 out per 1M
    c, exact = chat_cost_usd("glm/glm-5.3-flash", 1_000_000, 1_000_000)
    assert abs(c - 0.50) < 1e-9
    assert exact is True


def test_price_zero_tokens():
    c, _ = chat_cost_usd("glm/glm-5.3", 0, 0)
    assert c == 0.0


def test_fallback_price_flagged():
    p, exact = resolve("totally/unknown-model")
    assert exact is False
    c, exact2 = chat_cost_usd("totally/unknown-model", 1000, 1000)
    assert c > 0 and exact2 is False


def test_embed_price():
    c, _ = embed_cost_usd("gemini/gemini-embedding-2", 1_000_000)
    assert abs(c - 0.10) < 1e-9


def test_price_info_versioned():
    info: dict = price_info()
    assert info["version"] == PRICES_VERSION
    assert "glm/glm-5.3-flash" in info["models"]


# ---------------- metrics over synthetic spans --------------------------------
def _sp(dur_ms, status="ok", cost=0.001, feature="f", user="u", out_tok=50,
        cached=False):
    if cached:
        status, cost = "cached", 0.0
    return {"span_id": f"s{dur_ms}-{status}-{cost}-{out_tok}-{cached}-{feature}",
            "trace_id": "t1", "parent_id": None, "name": "llm_call",
            "kind": "llm_call", "ts": 1000.0, "dur_ms": dur_ms, "status": status,
            "error_type": None,
            "attrs": {"feature": feature, "user": user, "model": "m",
                      "prompt_tokens": 100, "completion_tokens": out_tok,
                      "cost_usd": cost, "cache_hit": "exact" if cached else None}}


def test_llm_metrics_cached_vs_live():
    spans = [_sp(100, cached=True) for _ in range(3)]
    spans += [_sp(500, cost=0.001) for _ in range(6)]
    spans += [_sp(2000, cost=0.002)]
    spans += [_sp(300, status="error", cost=0.0)]
    m = llm_metrics(spans)["*"]
    assert m["requests"] == 11
    assert m["cached"] == 3
    assert m["cache_hit_rate"] == 3 / 11
    assert m["live_calls"] == 8
    assert m["errors"] == 1
    assert m["error_rate"] == 1 / 8
    assert m["cost_usd"] == round(0.001 * 6 + 0.002, 6)
    lat = sorted([500.0] * 6 + [2000.0])
    assert m["latency_p50_ms"] == 500.0
    # p95 over 7 live-ok calls: ceil(0.95*7)=7 → max = 2000
    assert m["latency_p95_ms"] == 2000.0
    assert m["cost_per_1k_requests"] == pytest.approx(0.008 / 11 * 1000)


def test_llm_metrics_by_feature_bucket():
    spans = [_sp(10, feature="a", cost=0.001), _sp(20, feature="b", cost=0.002)]
    m = llm_metrics(spans, bucket_key="feature")
    assert set(m.keys()) == {"a", "b"}
    assert m["a"]["cost_usd"] == 0.001
    assert m["b"]["cost_usd"] == 0.002


def test_time_buckets():
    spans = ([_sp(10) for _ in range(3)] + [_sp(10) for _ in range(2)])
    for i, s in enumerate(spans):
        s["ts"] = 0.0 + i          # all within one 60s bucket
    rows = time_buckets(spans, bucket_s=60)
    assert len(rows) == 1
    assert rows[0]["reqs"] == 5
    assert rows[0]["cost_usd"] == round(0.001 * 5, 6)


# ---------------- attribution --------------------------------------------------
def test_attribution_rollup():
    spans = ([_sp(10, feature="chat", user="alice", cost=0.002)] * 2
             + [_sp(10, feature="sum", user="bob", cost=0.004, cached=True)])
    a = attribution(spans)
    assert a["total_cost_usd"] == round(0.002 * 2, 6)
    assert a["by_feature"]["chat"]["cost"] == 0.004
    assert a["by_feature"]["chat"]["requests"] == 2
    assert a["by_feature"]["sum"]["cached"] == 1
    assert a["by_user"]["alice"]["cost"] == 0.004
    assert a["by_user"]["bob"]["cached"] == 1
    assert a["total_requests"] == 3
    assert a["total_cached"] == 1


def test_attribution_by_model():
    spans = [_sp(10, cost=0.001), _sp(10, cost=0.001)]
    spans[1]["attrs"]["model"] = "other-model"
    a = attribution(spans)
    assert a["by_model"]["m"]["requests"] == 1
    assert a["by_model"]["other-model"]["requests"] == 1
