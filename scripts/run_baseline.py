#!/usr/bin/env python
"""PHASE 0 — baseline: trace-only (cache off), live provider calls.

Measures the un-optimized cost/latency of the workload and emits the alert
baseline (cost_per_1k, p95) that later phases are compared against.
Writes evals/results/baseline.json.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.driver import make_gateway, run_workload, summarize, write_json  # noqa: E402
from scripts.gen_workload import build_workload  # noqa: E402
from src import config  # noqa: E402


def main() -> int:
    n = int(os.environ.get("N", "300"))
    workers = int(os.environ.get("WORKERS", "4"))
    print(f"[baseline] n={n} model={config.TEXT_MODEL} cache=off live=1")
    gw = make_gateway(dry=False)
    wl = build_workload(n)
    for item in wl:
        item["cache"] = "off"          # baseline: no caching at all
    t0 = time.time()
    results = run_workload(gw, wl, workers=workers)
    s = summarize(results)
    s["phase"] = "baseline"
    s["model"] = config.TEXT_MODEL
    s["cache_mode"] = "off"
    s["wall_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(config.EVALS_DIR, "results", "baseline.json"), s)
    print("[baseline]", {k: s[k] for k in
                         ("requests", "errors", "cost_per_1k", "p50_ms",
                          "p95_ms", "hit_rate")})
    return 0 if s["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
