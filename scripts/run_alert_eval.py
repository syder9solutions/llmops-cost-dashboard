#!/usr/bin/env python
"""Alert precision eval — synthetic label-controlled scenarios (offline).

For each rule: N trials of "steady" traffic (no alert expected) and
"regression" traffic (alert expected). Reports fire-rate on regressions
(true positives) and on steady traffic (false alarms) per rule.

Scenarios inject synthetic spans directly into a temp store — no network,
deterministic seed. Writes evals/results/alert_eval.json.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import tempfile
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alerts import evaluate_alerts  # noqa: E402
from src.store import TraceStore  # noqa: E402

TRIALS = int(os.environ.get("TRIALS", "20"))
SEED = 20261213
BASELINE = {
    "error_rate": 0.10, "p95_ms": 2000.0, "p95_ratio": 1.5,
    "cost_per_1k": 3.0, "cost_ratio": 1.5, "psi_threshold": 0.30,
    "min_hit_rate": 0.05, "user_daily_budget": 0.50,
}


def synth_span(rng, *, status="ok", dur_ms=None, cost=None, out_tok=None,
               cached=False, user="u1", ts=1_000_000.0):
    if dur_ms is None:
        dur_ms = rng.gauss(900, 250)
    if cost is None:
        cost = rng.uniform(0.002, 0.004)
    if out_tok is None:
        out_tok = int(min(max(rng.gauss(120, 30), 40), 260))  # realistic band
    is_err = status not in ("ok",) and not cached
    attrs = {"feature": "chat_support", "user": user, "model": "m1",
             "prompt_tokens": 300,
             "completion_tokens": 0 if is_err else out_tok,
             "cost_usd": (0.0 if (cached or is_err) else cost)}
    return {"span_id": f"{rng.random():.17f}", "trace_id": "t", "parent_id": None,
            "name": "llm_call", "kind": "llm_call", "ts": ts, "dur_ms": max(1.0, dur_ms),
            "status": "cached" if cached else status, "error_type": None,
            "attrs": attrs}


def make_store(spans, tmpdir):
    st = TraceStore(os.path.join(tmpdir, "t.db"))
    st.add_spans(spans)
    return st


def scenario_steady(rng, n=120):
    # healthy cache-expecting traffic: ~35% cached spans
    spans = []
    for i in range(n):
        if rng.random() < 0.35:
            spans.append(synth_span(rng, cached=True, ts=1_000_000 + i))
        else:
            spans.append(synth_span(rng, ts=1_000_000 + i))
    return spans


def scenario_cache_collapse(rng, n=120):
    # cache stops hitting entirely (memcached dead / keyspace rotated)
    return [synth_span(rng, ts=1_000_000 + i) for i in range(n)]


def scenario_error_spike(rng, n=120):
    out = []
    for i in range(n):
        if rng.random() < 0.35:
            out.append(synth_span(rng, cached=True, ts=1_000_000 + i))
        else:
            out.append(synth_span(rng, status="error" if rng.random() < 0.4
                                  else "ok", ts=1_000_000 + i))
    return out


def scenario_latency_drift(rng, n=120):
    out = []
    for i in range(n):
        if rng.random() < 0.35:
            out.append(synth_span(rng, cached=True, ts=1_000_000 + i))
        else:
            out.append(synth_span(rng, dur_ms=rng.gauss(4200, 600),
                                  ts=1_000_000 + i))
    return out


def scenario_cost_spike(rng, n=120):
    # cost-per-call rises 4-5x (price change / verbose model) but total per-user
    # spend stays under the daily budget: 120 × ~0.013 ≈ $1.56 > $0.50 budget…
    # so lower the budget expectation: budget rule must NOT fire here → the
    # scenario sets budget=None via BASELINE override in SCENARIOS meta.
    out = []
    for i in range(n):
        if rng.random() < 0.35:
            out.append(synth_span(rng, cached=True, ts=1_000_000 + i))
        else:
            out.append(synth_span(rng, cost=rng.uniform(0.010, 0.016),
                                  ts=1_000_000 + i))
    return out


def scenario_output_drift(rng, n=240):
    # bimodal shift: long verbose answers appear (n=240 so ≥80 ok spans
    # survive the 35% cached mixing → PSI's sample floor)
    out = []
    for i in range(n):
        if rng.random() < 0.35:
            out.append(synth_span(rng, cached=True, ts=1_000_000 + i))
        else:
            verbose = rng.random() < 0.5
            out.append(synth_span(rng,
                                  out_tok=int(min(max(rng.gauss(420, 60), 300), 600)
                                              if verbose else
                                              max(rng.gauss(120, 30), 40)),
                                  ts=1_000_000 + i))
    return out


def scenario_budget_burn(rng, n=120):
    out = []
    for i in range(n):
        if rng.random() < 0.35:
            out.append(synth_span(rng, cached=True, user="whale",
                                  ts=1_000_000 + i))
        else:
            out.append(synth_span(rng, user="whale",
                                  cost=rng.uniform(0.012, 0.02),
                                  ts=1_000_000 + i))
    return out


SCENARIOS = {
    # (scenario_fn, expected_fires, baseline_overrides)
    "steady":        (scenario_steady,        {},                      {}),
    "error_spike":   (scenario_error_spike,   {"error_rate_jump": True}, {}),
    "latency_drift": (scenario_latency_drift, {"p95_latency_drift": True}, {}),
    # in a cost spike the burn rule would also legitimately fire; disable it
    # there to keep per-rule precision measurable in isolation
    "cost_spike":    (scenario_cost_spike,    {"cost_spike": True},
                     {"user_daily_budget": None}),
    "output_drift":  (scenario_output_drift,  {"output_drift": True}, {}),
    "budget_burn":   (scenario_budget_burn,   {"budget_burn": True}, {}),
    "cache_collapse": (scenario_cache_collapse, {"cache_hit_rate_collapse": True}, {}),
}

RULES = ["error_rate_jump", "p95_latency_drift", "cost_spike",
         "output_drift", "budget_burn", "cache_hit_rate_collapse"]


def main() -> int:
    rng = random.Random(SEED)
    expected_out_tok = [int(min(max(rng.gauss(120, 30), 40), 260))
                        for _ in range(400)]   # larger reference = stabler bins
    report = {"trials": TRIALS, "rules": {}, "scenarios": {}}

    # rule → {tp: fires_on_regression, fp: fires_on_steady}
    tp = {r: 0 for r in RULES}
    fp = {r: 0 for r in RULES}
    steady_out_samples = []

    for trial in range(TRIALS):
        for name, (fn, expected, overrides) in SCENARIOS.items():
            with tempfile.TemporaryDirectory() as td:
                spans = fn(rng)
                if name == "steady":
                    steady_out_samples.append(
                        [s["attrs"]["completion_tokens"] for s in spans])
                st = make_store(spans, td)
                baseline: Dict[str, Any] = dict(BASELINE)
                baseline["out_tokens_sample"] = expected_out_tok
                baseline["cache_expected"] = True   # steady state expects hits
                baseline.update(overrides)
                res = evaluate_alerts(st, now_ts=1_000_000 + 121 * 60,
                                      window_min=121 * 60, baseline=baseline)
                fires = res.get("firing", {})
                report["scenarios"].setdefault(name, []).append(
                    {r: bool(fires.get(r, False)) for r in RULES})
                for r in RULES:
                    should = bool(expected.get(r, False))
                    if should and fires.get(r, False):
                        tp[r] += 1
                    if not should and fires.get(r, False):
                        fp[r] += 1
                st.close()

    for r in RULES:
        report["rules"][r] = {
            "fire_rate_on_regression": round(tp[r] / TRIALS, 3),
            "false_alarm_rate_steady": round(fp[r] / (TRIALS * (len(SCENARIOS) - 1)), 4),
        }
    write_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "evals", "results", "alert_eval.json")
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    with open(write_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"wrote {write_path}")
    for r, v in report["rules"].items():
        print(f"  {r}: TP {v['fire_rate_on_regression']:.0%} "
              f"FA {v['false_alarm_rate_steady']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
