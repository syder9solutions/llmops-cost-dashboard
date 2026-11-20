"""QA regression: published README figures must match evals/report.json.

Root cause guarded: the hand-written intro prose ("the headline") drifted to
stale numbers (28-64% / 47%) while the auto-synced table and report said
84.4% / 60.3%, because sync_readme.py only regenerates the table between
markers. This test makes the prose and table fail loudly if they drift again.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "evals", "report.json")
README = os.path.join(ROOT, "README.md")
BEGIN = "<!-- BEGIN REPORT TABLE -->"
END = "<!-- END REPORT TABLE -->"


def _report():
    with open(REPORT) as f:
        return json.load(f)


def _readme():
    with open(README) as f:
        return f.read()


def test_report_has_headline_and_savings():
    r = _report()
    assert len(r["headline"]) >= 2
    assert r["savings_vs_baseline"] is not None
    assert r["savings_vs_baseline"]["cost_pct"] is not None
    assert r["savings_vs_baseline"]["p95_pct"] is not None


def test_readme_intro_headline_matches_report():
    """The prose headline figures must equal the report's savings."""
    r = _report()
    text = _readme()
    intro = text.split(BEGIN, 1)[0]  # only the hand-written prose
    savings = r["savings_vs_baseline"]
    cost = f"{savings['cost_pct']:g}"
    p95 = f"{savings['p95_pct']:g}"
    assert cost in intro, (
        f"README intro must state cost saving {cost}% from report.json")
    assert p95 in intro, (
        f"README intro must state p95 reduction {p95}% from report.json")


def test_readme_table_numbers_per_row():
    """Every auto-synced table row must match its report headline row."""
    r = _report()
    text = _readme()
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].splitlines()
    data_lines = [line for line in block
                  if line.startswith("|") and "---" not in line
                  and "Change" not in line]
    assert len(data_lines) == len(r["headline"])
    for line, row in zip(data_lines, r["headline"]):
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[0] == row["change"]
        assert cells[1] == f"${row['cost_per_1k_usd']:.4f}"
        assert cells[2] == f"{row['p95_ms']:.0f}"
        assert cells[3] == f"{row['cache_hit_rate']:.1%}"
        assert cells[4] == row["model"]


def test_report_arithmetic_self_consistent():
    """savings_vs_baseline must recompute from the headline rows."""
    r = _report()
    h = {x["change"]: x for x in r["headline"]}
    base = h["baseline (tracing only)"]
    last = h["+ semantic cache"]
    cost_pct = round((base["cost_per_1k_usd"] - last["cost_per_1k_usd"])
                     / base["cost_per_1k_usd"] * 100, 1)
    p95_pct = round((base["p95_ms"] - last["p95_ms"])
                    / base["p95_ms"] * 100, 1)
    assert cost_pct == r["savings_vs_baseline"]["cost_pct"]
    assert p95_pct == r["savings_vs_baseline"]["p95_pct"]
