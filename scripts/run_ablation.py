#!/usr/bin/env python
"""PHASE 2 — model/routing ablation: 3 strategies × 4 features, live.

Strategies:
  strong_only : every call → STRONG_MODEL
  economy     : short prompts → CHEAP_MODEL (cost-aware routing)
  blended     : billing_qa+chat_support → CHEAP, doc_summary+code_explain → STRONG

Measures $/1k and p95 per strategy with cache ON (semantic) — the honest
"after caching, what does routing buy?" question. Quality proxy: answer
length + spot-check; full quality eval belongs to the eval-suite project.
Writes evals/results/ablation.json.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.driver import make_gateway, run_workload, summarize, write_json  # noqa: E402
from scripts.gen_workload import build_workload  # noqa: E402
from src import config  # noqa: E402

N_PER_STRATEGY = int(os.environ.get("N", "84"))   # 3 strategies × 84 = 252 calls
STRATEGIES = ("strong_only", "economy", "blended")


def strategy_route(strategy: str, item: dict) -> dict:
    if strategy == "strong_only":
        item["route"] = "quality"
        item["model_override"] = config.STRONG_MODEL
    elif strategy == "economy":
        item["route"] = "economy"
    elif strategy == "blended":
        item["route"] = "quality"
        item["model_override"] = (
            config.CHEAP_MODEL if item["feature"] in ("chat_support", "billing_qa")
            else config.STRONG_MODEL)
    return item


def main() -> int:
    out = {"phase": "ablation", "n_per_strategy": N_PER_STRATEGY,
           "strategies": {}}
    for strat in STRATEGIES:
        print(f"[ablation] strategy={strat} n={N_PER_STRATEGY}")
        gw = make_gateway(dry=False, cache_mode="on")
        wl = build_workload(N_PER_STRATEGY, seed=20261212 + hash(strat) % 97)
        for item in wl:
            item = strategy_route(strat, item)
            item["cache"] = "on"
        t0 = time.time()
        # model_override: set request model per item
        results = []
        from concurrent.futures import ThreadPoolExecutor
        import threading

        def one(item):
            body = {"model": item.get("model_override") or config.TEXT_MODEL,
                    "messages": item["messages"],
                    "max_tokens": item.get("max_tokens", 120),
                    "feature": item["feature"], "user": item["user"],
                    "route": item.get("route", "quality"),
                    "cache": item.get("cache", "on")}
            t1 = time.perf_counter()
            try:
                resp = gw.chat_completion(body)
                wall = (time.perf_counter() - t1) * 1000.0
                tr = resp.get("tracing", {})
                return {"i": item["i"], "ok": True, "wall_ms": wall,
                        "cache": tr.get("cache"), "cost_usd": tr.get("cost_usd", 0.0),
                        "model": tr.get("model"),
                        "prompt_tokens": tr.get("prompt_tokens"),
                        "completion_tokens": tr.get("completion_tokens")}
            except Exception as e:  # noqa: BLE001
                return {"i": item["i"], "ok": False,
                        "wall_ms": (time.perf_counter() - t1) * 1000,
                        "error": f"{type(e).__name__}: {e}"}

        workers = int(os.environ.get("WORKERS", "2"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(one, wl):
                results.append(r)
        s = summarize(results)
        s["wall_s"] = round(time.time() - t0, 1)
        out["strategies"][strat] = s
        write_json(os.path.join(config.EVALS_DIR, "results", "ablation.json"), out)
        p95 = s["p95_ms"]
        print(f"[ablation] {strat}: ${s['cost_per_1k']:.3f}/1k "
              f"p95={(f'{p95:.0f}ms' if p95 is not None else 'n/a')} "
              f"hit={s['hit_rate']:.1%} err={s['errors']}")

    write_json(os.path.join(config.EVALS_DIR, "results", "ablation.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
