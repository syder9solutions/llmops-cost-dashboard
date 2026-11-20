"""FastAPI app: OpenAI-compatible gateway + observability API + dashboard."""
from __future__ import annotations

import statistics
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src import config, pricing
from src.alerts import evaluate_alerts
from src.applogic import Gateway
from src.cost import attribution, top_cost_features
from src.dash import render as render_dashboard
from src.metrics import llm_metrics, time_buckets
from src.store import TraceStore

app = FastAPI(title="LLMOps Cost & Latency Gateway", version="1.0.0")

_store: Optional[TraceStore] = None
_gw: Optional[Gateway] = None


def get_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore(config.TRACE_DB)
    return _store


def get_gateway() -> Gateway:
    global _gw
    if _gw is None:
        _gw = Gateway(get_store())
    return _gw


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # gateway extensions (attribution + cache control)
    feature: Optional[str] = None
    user: Optional[str] = None
    route: Optional[str] = None          # quality | economy
    cache: Optional[str] = None          # on | off | exact-only
    trace_id: Optional[str] = None

    model_config = {"extra": "forbid"}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    gw = get_gateway()
    payload = body.model_dump(exclude_none=True)
    try:
        return gw.chat_completion(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001 - map provider errors to 502
        raise HTTPException(status_code=502, detail=f"upstream: {type(e).__name__}: {e}")


@app.get("/stats")
async def stats(window_min: float = Query(120, ge=1, le=10000),
                bucket_s: float = Query(60, ge=5, le=3600)):
    store = get_store()
    import time as _t
    now = _t.time()
    spans = store.llm_spans(since_ts=now - window_min * 60)
    metrics = llm_metrics(spans)
    overheads = [s.get("attrs", {}).get("overhead_ms") for s in store.query(kind="request",
                                                                           limit=500)]
    overheads = [o for o in overheads if o is not None]
    a = attribution(spans)
    alerts = evaluate_alerts(store, now_ts=now, baseline=dict(config.ALERT_BASELINE))
    recent = store.query(kind="llm_call", limit=25)[-25:]
    recent_traces = [{
        "ts": r["ts"], "name": r["name"], "model": r["attrs"].get("model"),
        "dur_ms": r["dur_ms"], "status": r["status"],
        "cache_hit": r["attrs"].get("cache_hit"),
        "cost_usd": r["attrs"].get("cost_usd", 0.0),
    } for r in reversed(recent)]
    return {
        "window_min": window_min,
        "metrics": metrics,
        "time_buckets": time_buckets(spans[-400:], bucket_s=bucket_s),
        "top_features": top_cost_features(spans),
        "by_model": a["by_model"],
        "by_user": a["by_user"],
        "total_cost_usd": a["total_cost_usd"],
        "overhead_ms_p50": (statistics.median(overheads) if overheads else None),
        "cache_stats": get_gateway().cache.stats(),
        "prices_version": pricing.PRICES_VERSION,
        "alerts": alerts,
        "recent_traces": recent_traces,
    }


@app.get("/spans")
async def spans_api(kind: Optional[str] = None, trace_id: Optional[str] = None,
                    limit: int = Query(100, ge=1, le=1000)):
    return {"spans": get_store().query(kind=kind, trace_id=trace_id, limit=limit)}


@app.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    rows = get_store().query(trace_id=trace_id)
    if not rows:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"trace_id": trace_id, "spans": rows}


@app.get("/alerts")
async def alerts_api(window_min: float = 10.0):
    import time as _t
    return evaluate_alerts(get_store(), now_ts=_t.time(),
                           window_min=window_min,
                           baseline=dict(config.ALERT_BASELINE))


@app.get("/attribution")
async def attribution_api(window_min: float = 1440.0):
    import time as _t
    spans = get_store().llm_spans(since_ts=_t.time() - window_min * 60)
    return attribution(spans)


@app.get("/prices")
async def prices_api():
    return pricing.price_info()


@app.get("/health")
async def health():
    return {"ok": True, "spans": get_store().count()}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return render_dashboard()


@app.exception_handler(404)
async def not_found(request, exc):  # pragma: no cover
    return JSONResponse({"detail": "not found"}, status_code=404)
