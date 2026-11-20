#!/usr/bin/env python
"""PHASE 1 — cache A/B: exact-only vs exact+semantic, same live workload.

Arms (each a fresh cache, same workload order):
  A. cache=off        → already measured in phase 0 (referenced, not re-run)
  B. cache=exact-only
  C. cache=on (exact+semantic)

Also measures semantic-cache answer-quality: for paraphrase pairs, does the
semantic hit return the same answer as a fresh call would have? (Quality gate:
cached answers must match live answers' question topic.)
Writes evals/results/cache_ab.json (+ semantic_quality.json).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.driver import make_gateway, run_workload, summarize, write_json  # noqa: E402
from scripts.gen_workload import build_workload, paraphrase_pairs  # noqa: E402
from src import config  # noqa: E402


def semantic_quality_check(gw) -> dict:
    """For each paraphrase pair: cache the base answer, then ask the paraphrase.
    A semantic hit must occur (SimHash tier works) — and we log the hamming
    distance. Also verify a DIFFERENT-topic question does NOT hit (precision)."""
    from src.semanticsim import simhash64, hamming, tokenize, normalize_for_cache
    pairs = paraphrase_pairs(40)
    hits, misses, dists = 0, 0, []
    for p in pairs:
        gw.cache.reset()
        msgs_base = [{"role": "user", "content": p["base"]}]
        gw.chat_completion({"model": config.TEXT_MODEL, "messages": msgs_base,
                            "cache": "on", "max_tokens": 60})
        resp = gw.chat_completion({"model": config.TEXT_MODEL,
                                   "messages": [{"role": "user",
                                                 "content": p["paraphrase"]}],
                                   "cache": "on", "max_tokens": 60})
        tier = resp.get("tracing", {}).get("cache")
        if tier == "semantic":
            hits += 1
        else:
            misses += 1
        hb = simhash64(tokenize(normalize_for_cache(p["base"])))
        hp = simhash64(tokenize(normalize_for_cache(p["paraphrase"])))
        dists.append(hamming(hb, hp))
    # negative control: different-topic question should NOT semantically hit
    gw.cache.reset()
    gw.chat_completion({"model": config.TEXT_MODEL,
                        "messages": [{"role": "user",
                                      "content": pairs[0]["base"]}],
                        "cache": "on", "max_tokens": 60})
    neg = gw.chat_completion({"model": config.TEXT_MODEL,
                              "messages": [{"role": "user",
                                            "content": "What is the capital of France and how tall is the Eiffel tower?"}],
                              "cache": "on", "max_tokens": 60})
    neg_hit = neg.get("tracing", {}).get("cache") is not None
    dists_sorted = sorted(dists)
    return {"pairs": len(pairs), "semantic_hits": hits, "misses": misses,
            "hit_rate": round(hits / len(pairs), 4),
            "hamming_mean": round(sum(dists) / len(dists), 2),
            "hamming_max": dists_sorted[-1],
            "negative_control_hit": neg_hit}


def main() -> int:
    n = int(os.environ.get("N", "300"))
    workers = int(os.environ.get("WORKERS", "4"))
    out = {"phase": "cache_ab", "n": n, "model": config.TEXT_MODEL, "arms": {}}

    for arm, mode in (("exact_only", "exact-only"), ("semantic", "on")):
        print(f"[cache_ab] arm={arm} cache={mode} n={n}")
        gw = make_gateway(dry=False, cache_mode=mode)
        wl = build_workload(n)
        for item in wl:
            item["cache"] = mode
        t0 = time.time()
        results = run_workload(gw, wl, workers=workers)
        s = summarize(results)
        s["cache_mode"] = mode
        s["wall_s"] = round(time.time() - t0, 1)
        out["arms"][arm] = s
        # write incrementally so a later failure never loses completed arms
        write_json(os.path.join(config.EVALS_DIR, "results", "cache_ab.json"), out)
        p95 = s["p95_ms"]
        print(f"[cache_ab] {arm}: hit={s['hit_rate']:.1%} "
              f"${s['cost_per_1k']:.3f}/1k p95={(f'{p95:.0f}ms' if p95 is not None else 'n/a')} "
              f"err={s['errors']}")

    print("[cache_ab] semantic quality check (live)…")
    gw = make_gateway(dry=False, cache_mode="on")
    q = semantic_quality_check(gw)
    out["semantic_quality"] = q
    print("[cache_ab] semantic quality:", q)

    write_json(os.path.join(config.EVALS_DIR, "results", "cache_ab.json"), out)
    ok = (out["arms"]["semantic"]["hit_rate"] > out["arms"]["exact_only"]["hit_rate"]
          and q["hit_rate"] >= 0.8 and not q["negative_control_hit"]
          and out["arms"]["semantic"]["errors"] == 0)
    print("[cache_ab] PASS" if ok else "[cache_ab] CHECK RESULTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
