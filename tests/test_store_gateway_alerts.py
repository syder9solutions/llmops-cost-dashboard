"""Store, tracing, applogic (with dry client), alerts, and API tests — offline."""
import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.alerts import evaluate_alerts, psi  # noqa: E402
from src.applogic import Gateway  # noqa: E402
from src.cache import SemanticCache  # noqa: E402
from src.store import TraceStore  # noqa: E402
from src.tracing import Tracer, clear_sinks  # noqa: E402


# ---------------- store --------------------------------------------------------
def test_store_roundtrip(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    row = {"span_id": "s1", "trace_id": "t1", "parent_id": None,
           "name": "llm_call", "kind": "llm_call", "ts": 1.0, "dur_ms": 12.5,
           "status": "ok", "error_type": None,
           "attrs": {"cost_usd": 0.001, "model": "m"}}
    st.add_spans([row])
    got = st.query(kind="llm_call")
    assert len(got) == 1
    assert got[0]["attrs"]["cost_usd"] == 0.001
    assert got[0]["dur_ms"] == 12.5
    st.close()


def test_store_wal_and_indexes(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    mode = st._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    idx = st._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    names = {r[0] for r in idx}
    assert {"idx_spans_trace", "idx_spans_kind_ts"} <= names
    st.close()


def test_store_query_filters(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    for i in range(5):
        st.add_spans([{"span_id": f"s{i}", "trace_id": f"t{i%2}",
                       "parent_id": None, "name": "n", "kind": "llm_call",
                       "ts": float(i), "dur_ms": 1.0, "status": "ok",
                       "error_type": None, "attrs": {}}])
    assert st.count(kind="llm_call") == 5
    assert len(st.query(trace_id="t0")) == 3
    assert len(st.query(since_ts=3.0)) == 2
    assert len(st.query(until_ts=1.0)) == 2
    st.close()


def test_store_attrs_survive_json(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    st.add_spans([{"span_id": "s", "trace_id": "t", "name": "n", "kind": "k",
                   "ts": 0.0, "dur_ms": 0.0, "status": "ok",
                   "attrs": {"nested": {"a": [1, 2]}}}])
    got = st.query()[0]
    assert got["attrs"]["nested"]["a"] == [1, 2]
    st.close()


# ---------------- tracing ------------------------------------------------------
def test_span_tree_and_timing():
    clear_sinks()
    captured = []
    from src.tracing import add_sink
    add_sink(captured.append)
    tr = Tracer()
    with tr.span("request", kind="request") as root:
        with tr.span("llm_call", kind="llm_call") as child:
            child.attrs["cost_usd"] = 0.001
        with tr.span("cache_lookup", kind="cache") as c2:
            pass
    assert root.trace_id == child.trace_id
    assert child.parent_id == root.span_id
    assert c2.parent_id == root.span_id
    assert root.dur_ms >= child.dur_ms
    assert len(captured) == 3
    # spans close inner-first: llm_call and cache_lookup persist before request
    by_name = {s.name: s for s in captured}
    assert by_name["llm_call"].attrs["cost_usd"] == 0.001
    clear_sinks()


def test_span_error_captured():
    clear_sinks()
    captured = []
    from src.tracing import add_sink
    add_sink(captured.append)
    tr = Tracer()
    with pytest.raises(RuntimeError):
        with tr.span("boom", kind="request"):
            raise RuntimeError("x")
    sp = captured[0]
    assert sp.status == "error" and sp.error_type == "RuntimeError"
    clear_sinks()


def test_overhead_attr_present():
    clear_sinks()
    captured = []
    from src.tracing import add_sink
    add_sink(captured.append)
    tr = Tracer()
    with tr.span("x", kind="request"):
        pass
    ov = captured[0].attrs.get("overhead_ms")
    assert ov is not None and ov < 50  # must be tiny
    clear_sinks()


# ---------------- gateway with dry client -------------------------------------
class DryClient:
    def __init__(self):
        self.n = 0

    def chat(self, model, messages, max_tokens=None, temperature=0.2, extra=None):
        self.n += 1
        return {"id": f"d{self.n}", "model": model,
                "choices": [{"message": {"role": "assistant",
                            "content": f"answer {self.n}"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30,
                          "total_tokens": 80}}


def _gw(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    return Gateway(st, client=DryClient(), cache=SemanticCache()), st  # type: ignore[arg-type]


def test_gateway_live_call_span_attrs(tmp_path):
    gw, st = _gw(tmp_path)
    resp = gw.chat_completion({"model": "glm/glm-5.3-flash",
                               "messages": [{"role": "user", "content": "q1"}],
                               "feature": "f", "user": "u"})
    time.sleep(0.05)  # allow async sink… sink is sync; just read
    spans = st.query(kind="llm_call")
    assert len(spans) == 1
    a = spans[0]["attrs"]
    assert a["prompt_tokens"] == 50 and a["completion_tokens"] == 30
    assert a["cost_usd"] > 0
    assert resp["tracing"]["cost_usd"] == a["cost_usd"]
    st.close()


def test_gateway_exact_cache_hit(tmp_path):
    gw, st = _gw(tmp_path)
    body = {"model": "glm/glm-5.3-flash",
            "messages": [{"role": "user", "content": "same q"}],
            "feature": "f", "user": "u"}
    r1 = gw.chat_completion(dict(body))
    r2 = gw.chat_completion(dict(body))
    assert r2["tracing"]["cache"] == "exact"
    assert r2["tracing"]["cost_usd"] == 0.0
    cached_spans = st.query(kind="llm_call", status="cached")
    assert len(cached_spans) == 1
    st.close()


def test_gateway_cache_off(tmp_path):
    gw, st = _gw(tmp_path)
    body = {"model": "glm/glm-5.3-flash",
            "messages": [{"role": "user", "content": "same q"}],
            "feature": "f", "user": "u", "cache": "off"}
    gw.chat_completion(dict(body))
    gw.chat_completion(dict(body))
    assert st.count(kind="llm_call") == 2
    assert len(st.query(kind="llm_call", status="cached")) == 0
    st.close()


def test_gateway_economy_routing(tmp_path):
    gw, st = _gw(tmp_path)
    gw.chat_completion({"model": "glm/glm-5.3",  # expensive requested
                        "messages": [{"role": "user", "content": "hi"}],
                        "route": "economy"})
    spans = st.query(kind="llm_call")
    assert spans[0]["attrs"]["model"] == config.CHEAP_MODEL
    st.close()


def test_gateway_banned_model_rejected(tmp_path):
    gw, st = _gw(tmp_path)
    with pytest.raises(ValueError):
        gw.chat_completion({"model": "antigravity/gemini-3-flash",
                            "messages": [{"role": "user", "content": "x"}]})
    st.close()


def test_gateway_empty_messages_rejected(tmp_path):
    gw, st = _gw(tmp_path)
    with pytest.raises(ValueError):
        gw.chat_completion({"model": "m", "messages": []})
    st.close()


def test_gateway_semantic_hit_zero_cost(tmp_path):
    gw, st = _gw(tmp_path)
    base = "What are the current rate limits for the paid API tier and how are they calculated?"
    para = ("Hey, quick question — what are the current rate limits for the paid "
            "API tier and how are they calculated?")
    gw.chat_completion({"model": "glm/glm-5.3-flash",
                        "messages": [{"role": "user", "content": base}]})
    r = gw.chat_completion({"model": "glm/glm-5.3-flash",
                            "messages": [{"role": "user", "content": para}]})
    assert r["tracing"]["cache"] == "semantic"
    assert r["tracing"]["cost_usd"] == 0.0
    st.close()


# ---------------- alerts -------------------------------------------------------
def test_psi_identical():
    a = [float(x) for x in range(1, 101)]
    v = psi(a, a)
    assert v is not None and v < 0.05


def test_psi_shifted():
    a = [float(x) for x in range(1, 101)]
    b = [float(x) + 500 for x in range(1, 101)]
    v = psi(a, b)
    assert v is not None and v > 1.0


def test_psi_too_few():
    assert psi([1.0] * 5, [1.0] * 5) is None


def _synth(st, n=50, status="ok", dur=900.0, cost=0.003, out_tok=120, user="u1"):
    for i in range(n):
        st.add_spans([{"span_id": f"s{i}", "trace_id": "t", "parent_id": None,
                       "name": "llm_call", "kind": "llm_call",
                       "ts": 1_000_000.0 + i, "dur_ms": dur, "status": status,
                       "error_type": None,
                       "attrs": {"feature": "f", "user": user, "model": "m",
                                 "prompt_tokens": 300,
                                 "completion_tokens": out_tok,
                                 "cost_usd": cost}}])


def test_alert_error_jump_fires(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=30, status="error")
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"error_rate": 0.10})
    assert res["rules"]["error_rate_jump"]["firing"] is True
    st.close()


def test_alert_steady_quiet(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    # steady traffic WITH caching armed: ~40% of requests served from cache
    # (the healthy state of a cache-expecting deployment)
    for i in range(30):
        _synth(st, n=1)
    for i in range(20):
        st.add_spans([{"span_id": f"c{i}", "trace_id": "t", "parent_id": None,
                       "name": "llm_call", "kind": "llm_call",
                       "ts": 1_000_040.0 + i, "dur_ms": 4.0, "status": "cached",
                       "error_type": None, "attrs": {"feature": "f", "user": "u1",
                                                     "model": "m", "cost_usd": 0.0}}])
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"error_rate": 0.10, "p95_ms": 2000.0,
                                    "p95_ratio": 1.5, "cost_per_1k": 3.0,
                                    "cost_ratio": 1.5, "min_hit_rate": 0.05,
                                    "cache_expected": True,
                                    "user_daily_budget": 0.50})
    fires = {k: v["firing"] for k, v in res["rules"].items()}
    assert not any(fires.values()), fires
    st.close()


def test_alert_cache_collapse_fires_only_when_expected(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    # 30 healthy + 25 cached → healthy hit-rate ~45%
    for i in range(30):
        _synth(st, n=1)
    for i in range(25):
        st.add_spans([{"span_id": f"c{i}", "trace_id": "t", "parent_id": None,
                       "name": "llm_call", "kind": "llm_call",
                       "ts": 1_000_050.0 + i, "dur_ms": 5.0, "status": "cached",
                       "error_type": None, "attrs": {"feature": "f", "user": "u",
                                                     "model": "m", "cost_usd": 0.0}}])
    res_ok = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                             baseline={"min_hit_rate": 0.05,
                                       "cache_expected": True})
    assert res_ok["rules"]["cache_hit_rate_collapse"]["firing"] is False
    st2 = TraceStore(str(tmp_path / "t2.db"))
    _synth(st2, n=50)  # zero cached
    res_bad = evaluate_alerts(st2, now_ts=1_000_060, window_min=1,
                              baseline={"min_hit_rate": 0.05,
                                        "cache_expected": True})
    assert res_bad["rules"]["cache_hit_rate_collapse"]["firing"] is True
    # without cache_expected the rule is not even armed
    res_off = evaluate_alerts(st2, now_ts=1_000_060, window_min=1,
                              baseline={"min_hit_rate": 0.05})
    assert "cache_hit_rate_collapse" not in res_off["rules"]
    st.close(); st2.close()


def test_alert_latency_drift_fires(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=50, dur=5000.0)
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"p95_ms": 2000.0, "p95_ratio": 1.5})
    assert res["rules"]["p95_latency_drift"]["firing"] is True
    st.close()


def test_alert_cost_spike_fires(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=50, cost=0.02)     # 20/1k >> baseline 3/1k
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"cost_per_1k": 3.0, "cost_ratio": 1.5})
    assert res["rules"]["cost_spike"]["firing"] is True
    st.close()


def test_alert_budget_burn_fires(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=50, cost=0.05, user="whale")   # $2.50 total > $0.50 budget
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"user_daily_budget": 0.50})
    assert res["rules"]["budget_burn"]["firing"] is True
    st.close()


def test_alert_drift_psi_fires(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=100, out_tok=400)   # ≥80 ok spans → above PSI sample floor
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"psi_threshold": 0.25,
                                    "out_tokens_sample": [float(x) for x in
                                                          range(80, 160)]})
    assert res["rules"]["output_drift"]["firing"] is True
    st.close()


def test_psi_sample_floor(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=50, out_tok=400)    # below the 80-sample floor → rule unarmed
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1,
                          baseline={"psi_threshold": 0.25,
                                    "out_tokens_sample": [float(x) for x in
                                                          range(80, 160)]})
    assert "output_drift" not in res["rules"]
    st.close()


def test_alert_min_requests_guard(tmp_path):
    st = TraceStore(str(tmp_path / "t.db"))
    _synth(st, n=5, status="error")
    res = evaluate_alerts(st, now_ts=1_000_060, window_min=1, baseline={})
    assert "error_rate_jump" not in res["rules"]   # guard: n>=10
    st.close()


# ---------------- API layer (fastapi TestClient, dry) ---------------------------
def test_api_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DB", str(tmp_path / "t.db"))
    # fresh import so config picks up env
    for mod in list(sys.modules):
        if mod.startswith(("src", "app")):
            del sys.modules[mod]
    import importlib
    import src.config as cfg
    importlib.reload(cfg)
    cfg.TRACE_DB = str(tmp_path / "t.db")
    import app.main as main
    importlib.reload(main)
    main._store = None
    main._gw = None

    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True

    r = client.get("/dashboard")
    assert r.status_code == 200 and "LLMOps" in r.text

    r = client.get("/prices")
    assert r.status_code == 200 and "glm/glm-5.3-flash" in r.json()["models"]

    r = client.get("/traces/does-not-exist")
    assert r.status_code == 404

    r = client.post("/v1/chat/completions", json={
        "model": "antigravity/x", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 422

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    # contract: either a 200 with full tracing metadata (live gateway is up)
    # or a clean upstream error — never a crash or silent body
    if r.status_code == 200:
        tr = r.json().get("tracing", {})
        assert tr.get("trace_id") and "cost_usd" in tr
    else:
        assert r.status_code in (422, 502)
    clear_sinks()
