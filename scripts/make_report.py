#!/usr/bin/env python
"""Aggregate evals/results/*.json → evals/report.json + printed README table
(the headline: $/1k, p95, hit-rate per optimization stage).

Data-quality gate: an arm with zero successful calls (errors == requests) is
NOT a valid measurement — it is excluded from the headline and recorded in
`excluded` with the reason. This prevents an exhausted-quota run from being
reported as a real "100% cost saving".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402

R = config.EVALS_DIR


def load(name):
    p = os.path.join(R, "results", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def valid(s):
    """A stage is reportable only if it made real, successful calls."""
    if not s:
        return False
    errs = s.get("errors", 0)
    live = s.get("live_calls", 0)
    req = s.get("requests", 0)
    # baseline reports live_calls; arms report requests-errors implicitly
    successes = live if live else (req - errs)
    return successes > 0 and s.get("cost_per_1k") is not None


def main() -> int:
    baseline = load("baseline.json")
    ab = load("cache_ab.json")
    abl = load("ablation.json")
    alerts = load("alert_eval.json")

    rows, excluded = [], []

    def add(label, s, model, note=""):
        if valid(s):
            rows.append((label, s["cost_per_1k"], s.get("p95_ms"),
                         s.get("hit_rate", 0.0), model))
        elif s:
            excluded.append({"change": label, "reason": "no successful calls",
                             "errors": s.get("errors"), "requests": s.get("requests")
                             or s.get("live_calls"), "note": note})

    if baseline:
        add("baseline (tracing only)", baseline, baseline.get("model", "-"))
    if ab:
        for arm, label in (("exact_only", "+ exact cache"),
                           ("semantic", "+ semantic cache")):
            s = ab["arms"].get(arm)
            if s:
                add(label, s, ab.get("model", "-"),
                    f"cache_mode={s.get('cache_mode')}")
    if abl:
        s = (abl["strategies"].get("blended") or abl["strategies"].get("economy"))
        if s:
            add("+ cheap-model routing", s, "blended",
                "model tiers converged on gemini due to gateway quota")

    report = {
        "headline": [
            {"change": c, "cost_per_1k_usd": round(cst, 4),
             "p95_ms": round(p, 0) if p is not None else None,
             "cache_hit_rate": round(h, 3), "model": m}
            for c, cst, p, h, m in rows
        ],
        "excluded": excluded,
        "comparability_note": (
            "Baseline ran on groq/openai/gpt-oss-120b; cache arms ran on "
            "gemini/gemini-3.1-flash-lite after the shared gateway's groq org "
            "quota was exhausted. Both use exact per-model pricing from "
            "src/pricing.py, but cross-stage cost deltas mix model tiers: "
            "treat the cache hit-rate and p95 changes as the primary evidence, "
            "and the $/1k delta as indicative only."
        ),
        "savings_vs_baseline": None,
        "alert_rules": (alerts or {}).get("rules"),
        "semantic_quality": (ab or {}).get("semantic_quality"),
    }
    if len(rows) >= 2 and rows[0][1]:
        first, last = rows[0], rows[-1]
        report["savings_vs_baseline"] = {
            "cost_pct": round((first[1] - last[1]) / first[1] * 100.0, 1),
            "p95_pct": (round((first[2] - last[2]) / first[2] * 100.0, 1)
                        if first[2] and last[2] else None),
        }

    out_path = os.path.join(R, "report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}\n")
    print(f"{'change':<24}{'$/1k':>10}{'p95 ms':>10}{'hit-rate':>10}  model")
    for c, cst, p, h, m in rows:
        ps = f"{p:.0f}" if p is not None else "n/a"
        print(f"{c:<24}{cst:>10.3f}{ps:>10}{h:>10.1%}  {m}")
    for e in excluded:
        print(f"EXCLUDED {e['change']}: {e['reason']} "
              f"(errors={e['errors']}/{e['requests']}) {e.get('note','')}")
    if report["savings_vs_baseline"]:
        s = report["savings_vs_baseline"]
        print(f"\nvs baseline: cost −{s['cost_pct']}%  p95 −{s.get('p95_pct')}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
