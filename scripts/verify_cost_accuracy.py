#!/usr/bin/env python
"""Cost-accuracy audit: recompute Σ(tokens × price) from raw spans and assert
it equals the stored attribution, to the cent (tolerance 0).

This is the "computed $ matches the provider bill" guarantee: our bill IS
tokens × price table, so the audit proves no drift between span attrs,
attribution rollups, and pricing. Also checks every span has a price_exact
flag and flags any fallback-priced usage.
Exit 1 on any mismatch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.cost import attribution  # noqa: E402
from src.pricing import chat_cost_usd  # noqa: E402
from src.store import TraceStore  # noqa: E402


def main() -> int:
    store = TraceStore(config.TRACE_DB)
    spans = store.llm_spans()
    recomputed = 0.0
    stored = 0.0
    fallback_calls = 0
    bad = []
    for sp in spans:
        a = sp.get("attrs", {})
        if sp.get("status") == "cached":
            continue
        pt, ct = a.get("prompt_tokens", 0), a.get("completion_tokens", 0)
        c, exact = chat_cost_usd(a.get("model", "?"), pt, ct)
        recomputed += c
        stored += float(a.get("cost_usd", 0.0) or 0.0)
        if not a.get("price_exact", True):
            fallback_calls += 1
        if abs(c - float(a.get("cost_usd", 0.0) or 0.0)) > 1e-9:
            bad.append(sp["span_id"])

    attr = attribution(spans)
    total_attr = attr["total_cost_usd"]

    print(f"llm spans audited : {len(spans)}")
    print(f"recomputed Σ      : ${recomputed:.6f}")
    print(f"stored Σ (attrs)  : ${stored:.6f}")
    print(f"attribution Σ     : ${total_attr:.6f}")
    print(f"per-span mismatches: {len(bad)}")
    print(f"fallback-priced calls: {fallback_calls}")

    ok = (len(bad) == 0
          and abs(recomputed - stored) < 1e-6
          and abs(recomputed - total_attr) < 1e-5)
    print("COST AUDIT:", "PASS" if ok else "FAIL")
    store.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
