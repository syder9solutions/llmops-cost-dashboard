#!/usr/bin/env python
"""Inject the live headline table from evals/report.json into README markers.

Idempotent: the block between <!-- BEGIN REPORT TABLE --> / <!-- END REPORT TABLE -->
is replaced in place. Run after scripts/make_report.py.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
REPORT = os.path.join(ROOT, "evals", "report.json")
BEGIN = "<!-- BEGIN REPORT TABLE -->"
END = "<!-- END REPORT TABLE -->"


def main() -> int:
    if not os.path.exists(REPORT):
        print(f"missing {REPORT}; run scripts/make_report.py first", file=sys.stderr)
        return 1
    with open(REPORT) as f:
        report = json.load(f)
    rows = report.get("headline", [])
    lines = [BEGIN, "| Change | $/1k req | p95 latency (ms) | cache hit-rate | model |",
             "|---|---|---|---|---|"]
    for r in rows:
        p = r.get("p95_ms")
        p95 = f"{p:.0f}" if p is not None else "n/a"
        lines.append(
            f"| {r['change']} | ${r['cost_per_1k_usd']:.4f} | {p95} | "
            f"{r['cache_hit_rate']:.1%} | {r.get('model', '-')} |")
    lines.append(END)
    block = "\n".join(lines)

    with open(README) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        print("README markers missing", file=sys.stderr)
        return 1
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    new = pre + block + post
    with open(README, "w") as f:
        f.write(new)
    print(f"synced {len(rows)} rows into README")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
